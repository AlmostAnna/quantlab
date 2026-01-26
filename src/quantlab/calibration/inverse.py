"""
Inverse problems module.

This module contains implementation of Heston model parameters recovery.
"""
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
from py_vollib.black.implied_volatility import (
    implied_volatility_of_undiscounted_option_price,
)
from scipy.optimize import differential_evolution, minimize

from quantlab.calibration.utils import (
    safe_implied_volatility_of_undiscounted_price as safe_iv_of_undiscounted_price,
)
from quantlab.market_data.market_state import MarketState

# Vectorize for efficiency
implied_volatility_vec = np.vectorize(implied_volatility_of_undiscounted_option_price)


def recover_heston_params_from_prices(
    strikes: np.ndarray,
    maturities: np.ndarray,
    prices: np.ndarray,
    forward: float,
    discount_factors: np.ndarray,
    initial_guess: Dict[str, float],
    pricing_func: Callable,
    pricing_kwargs: Dict[str, Any],
    bounds: Dict[
        str, Tuple[float, float]
    ],  # Bounds are crucial for global optimizers like DE
    weights: np.ndarray = None,  # Optional weights
    method: Union[str, Callable] = "L-BFGS-B",  # Allow string name or direct function
    optimizer_options: Dict[
        str, Any
    ] = None,  # Pass options specific to the chosen optimizer
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Recover Heston parameters from a set of observed option prices.

    Parameters:
    - strikes: (N,) array of strike prices.
    - maturities: (N,) array of times to maturity.
    - prices: (N,) array of observed undiscounted option prices.
    - forward: Scalar or (N,) array of forward prices.
    - discount_factors: (N,) array of discount factors.
    - initial_guess: Dict of initial parameter values {'v0': ..., 'kappa': ..., ...}.
                     Used as the starting point for local optimizers like L-BFGS-B.
    - pricing_func: A function that takes (S, K, T, r, q, **params) and returns price.
    - pricing_kwargs: Additional keyword arguments for the pricing function.
    - bounds: Dict of (lower, upper) bounds for each parameter
            (required for differential_evolution),e.g.,
            {'kappa': (0.1, 10.0), 'theta': (0.01, 0.2), ...}
    - weights: Optional (N,) array to weight the objective function.
                If provided, objective becomes:
                sum(weights * (model_iv - target_iv)^2) / sum(weights).
                Useful for vega-weighted calibration or emphasizing liquid options.
                Default: None (uniform weights).
    - method: Name of the scipy optimizer ('L-BFGS-B', 'differential_evolution')
            or the optimizer function itself.
            'L-BFGS-B' is fast but local. 'differential_evolution' is slower but global.
    - optimizer_options: Dictionary of options passed directly
                        to the scipy optimizer function.
                        e.g., {'maxiter': 1000, 'popsize': 15} for
                        differential_evolution.
                        e.g., {'ftol': 1e-9} for L-BFGS-B.
                        Defaults will be used if None.
    - verbose: Print optimization status messages.

    Returns:
    - Dict of recovered parameters.
    """
    # --- 1. Validate input arrays have same length ---
    assert (
        len(strikes) == len(maturities) == len(prices) == len(discount_factors)
    ), f"Array lengths don't match: {len(strikes)}, {len(maturities)}, {len(prices)}, {len(discount_factors)}"  # noqa: E501

    # Handle forward as scalar or array properly
    if np.isscalar(forward):
        forward_array = np.full_like(strikes, forward)
    else:
        forward_array = forward
        assert len(forward_array) == len(
            strikes
        ), f"Forward array length mismatch: {len(forward_array)} vs {len(strikes)}"

    # --- 1. Convert observed prices to implied vols  ---
    undiscounted_prices = prices / discount_factors
    intrinsic = np.maximum(forward - strikes, 0.0)
    undiscounted_prices_safe = np.maximum(undiscounted_prices, intrinsic + 1e-12)
    observed_ivs = implied_volatility_vec(
        undiscounted_prices_safe, forward, strikes, maturities, "c"
    )

    _debug_counter = 0

    # --- 2. Define objective function ---
    def objective(params_vec: np.ndarray) -> float:
        nonlocal _debug_counter
        # Map flat vector back to dict
        param_names = list(initial_guess.keys())
        current_params_dict = {name: val for name, val in zip(param_names, params_vec)}

        if verbose and _debug_counter < 5:
            print(f"DEBUG: Objective called with params: {current_params_dict}")
            _debug_counter += 1

        model_ivs = []
        for i in range(len(strikes)):
            try:
                # S_val = forward if np.isscalar(forward) else forward[i]
                S_val = forward_array[i]  # Use the corrected forward array
                K_val = strikes[i]
                T_val = maturities[i]

                if verbose and _debug_counter <= 5:
                    print(
                        f"  DEBUG: Pricing S={S_val:.2f}, K={K_val:.2f}, T={T_val:.2f}"
                    )

                model_price = pricing_func(
                    current_params_dict,  # the params dictionary
                    S_val,  # S
                    K_val,  # K
                    T_val,  # T
                )

                if verbose and _debug_counter <= 5:
                    print(f"    DEBUG: Returned price = {model_price}")

                undisc_model_price = model_price / discount_factors[i]
                intrinsic_model = max(S_val - K_val, 0.0)
                undisc_model_price_safe = max(
                    undisc_model_price, intrinsic_model + 1e-12
                )
                model_iv = safe_iv_of_undiscounted_price(
                    undisc_model_price_safe, S_val, K_val, T_val, "c"
                )
                if verbose and _debug_counter <= 5:
                    print(
                        f"    DEBUG: Calculated IV = {model_iv:.6f}, Observed IV = {observed_ivs[i]:.6f}"  # noqa: E501
                    )

                model_ivs.append(model_iv)
            except Exception as e:
                if verbose:
                    print(
                        f"Warning: Invalid params {current_params_dict} caused error: {e}. Returning large error."  # noqa: E501
                    )
                return 1e6  # Penalize invalid regions heavily

        model_ivs = np.array(model_ivs)
        error = observed_ivs - model_ivs

        if weights is not None:
            weighted_mse = np.average(error**2, weights=weights)
            return weighted_mse
        else:
            mse = np.mean(error**2)
            return mse

    # --- 3. Set up optimizer ---
    x0 = list(initial_guess.values())  # Initial guess vector for local optimizers
    bounds_list = [
        bounds[name] for name in initial_guess.keys()
    ]  # Bounds must be a list of tuples for DE

    optimizer_opts = optimizer_options or {}  # Use provided options or empty dict
    if verbose:
        optimizer_opts["disp"] = True  # Add display option if verbose

    # --- 4. Run optimization based on method ---
    if method == "differential_evolution":
        if verbose:
            print("Using Differential Evolution (global optimizer)...")
        # Note: DE doesn't use x0, it uses bounds for population initialization
        result = differential_evolution(objective, bounds_list, **optimizer_opts)
    elif method == "L-BFGS-B":
        if verbose:
            print("Using L-BFGS-B (local optimizer)...")
        # Note: L-BFGS-B uses x0 as the starting point
        result = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds_list, options=optimizer_opts
        )
    else:
        # For more flexibility, allow passing the optimizer function directly
        # This requires careful handling of arguments expected by different optimizers
        if callable(method):
            raise NotImplementedError(
                "Passing custom optimizer functions directly is not yet fully implemented. Use 'L-BFGS-B' or 'differential_evolution'."  # noqa: E501
            )
        else:
            # Assume it's a string for a method supported by scipy.optimize.minimize
            result = minimize(
                objective, x0, method=method, bounds=bounds_list, options=optimizer_opts
            )

    # --- 5. Check success and return ---
    if not result.success:
        print(f"Optimization status: {result.message}")
        # Check if we got a reasonably good solution
        # despite not meeting convergence criteria
        if result.fun < 1e-3:  # If MSE is still reasonably small
            print(f"Accepting suboptimal solution with MSE={result.fun:.6f}")
            recovered_params = {
                name: val for name, val in zip(list(initial_guess.keys()), result.x)
            }
            return recovered_params
        else:
            # Check for common termination reasons that aren't true failures
            msg_lower = str(result.message).lower() if result.message else ""

            # Common DE messages about iteration limits
            de_limits = any(
                word in msg_lower
                for word in ["iteration", "maxiter", "max evaluations"]
            )
            # Common minimize messages about iteration/function eval limits
            min_limits = any(
                word in msg_lower
                for word in ["iterations", "maxfev", "maxiter", "function evaluation"]
            )
            # Convergence tolerance reached (but not perfect)
            tolerance_reached = any(
                word in msg_lower for word in ["ftol", "xtol", "gtol", "converged"]
            )

            if de_limits or min_limits or tolerance_reached:
                print(
                    f"Optimization terminated (likely reached limits), returning best result (MSE={result.fun:.6f})"  # noqa: E501
                )
                recovered_params = {
                    name: val for name, val in zip(list(initial_guess.keys()), result.x)
                }
                return recovered_params
            else:
                # Genuine failure (constraints violated, numerical issues, etc.)
                print("Returning initial guess due to optimization failure.")
                return initial_guess
    else:
        # Success - converged properly
        recovered_params = {
            name: val for name, val in zip(list(initial_guess.keys()), result.x)
        }
        return recovered_params


def recover_heston_params_from_implied_vols(
    strikes: np.ndarray,
    maturities: np.ndarray,
    target_implied_vols: np.ndarray,
    market_state: MarketState,
    initial_guess: Dict[str, float],
    pricing_func: Callable,  # Your cos_wrapper
    pricing_kwargs: Dict[str, Any],
    bounds: Dict[str, Tuple[float, float]],
    weights: np.ndarray = None,
    method: Union[str, Callable] = "L-BFGS-B",
    optimizer_options: Dict[str, Any] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Recover Heston parameters from implied volatility surface.

    Parameters:
    - strikes: (N,) array of strike prices.
    - maturities: (N,) array of times to maturity.
    - target_implied_vols: (N,) array of observed IVs.
    - market_state: State of the market.
    - initial_guess: Dict of initial parameter values {'v0': ..., 'kappa': ..., ...}.
                     Used as the starting point for local optimizers like L-BFGS-B.
    - pricing_func: A function that takes (S, K, T, r, q, **params) and returns price.
    - pricing_kwargs: Additional keyword arguments for the pricing function.
    - bounds: Dict of (lower, upper) bounds for each parameter
            (required for differential_evolution),e.g.,
            {'kappa': (0.1, 10.0), 'theta': (0.01, 0.2), ...}
    - weights: Optional (N,) array to weight the objective function.
                If provided, objective becomes:
                sum(weights * (model_iv - target_iv)^2) / sum(weights).
                Useful for vega-weighted calibration or emphasizing liquid options.
                Default: None (uniform weights).
    - method: Name of the scipy optimizer ('L-BFGS-B', 'differential_evolution')
            or the optimizer function itself.
            'L-BFGS-B' is fast but local. 'differential_evolution' is slower but global.
    - optimizer_options: Dictionary of options passed directly
                        to the scipy optimizer function.
                        e.g., {'maxiter': 1000, 'popsize': 15} for
                        differential_evolution.
                        e.g., {'ftol': 1e-9} for L-BFGS-B.
                        Defaults will be used if None.
    - verbose: Print optimization status messages.

    Returns:
    - Dict of recovered parameters.
    """
    # Validate input arrays
    assert (
        len(strikes) == len(maturities) == len(target_implied_vols)
    ), f"Array lengths don't match: {len(strikes)}, {len(maturities)}, {len(target_implied_vols)}"  # noqa: E501

    # Extract market parameters
    S0 = market_state.stock_price
    r = market_state.interest_rate
    q = getattr(market_state, "dividend_yield", 0.0)
    t_now = getattr(market_state, "time", 0.0)

    # Calculate forwards and discount factors
    if np.isscalar(maturities):
        forward_array = np.full_like(
            strikes, S0 * np.exp((r - q) * (maturities - t_now))
        )
        discount_factors = np.exp(-r * (maturities - t_now))
    else:
        forward_array = S0 * np.exp((r - q) * (maturities - t_now))
        discount_factors = np.exp(-r * (maturities - t_now))

    _debug_counter = 0

    def objective(params_vec: np.ndarray) -> float:
        nonlocal _debug_counter
        # Map flat vector back to dict
        param_names = list(initial_guess.keys())
        current_params_dict = {name: val for name, val in zip(param_names, params_vec)}

        if verbose and _debug_counter < 5:
            print(f"DEBUG: Objective called with params: {current_params_dict}")
            _debug_counter += 1

        model_implied_vols = []
        for i in range(len(strikes)):
            try:
                S_val = forward_array[i]  # Use forward from array
                K_val = strikes[i]
                T_val = maturities[i] - t_now  # Time to expiry

                if verbose and _debug_counter <= 5:
                    print(
                        f"  DEBUG: Pricing S={S_val:.2f}, K={K_val:.2f}, T={T_val:.2f}"
                    )

                model_price = pricing_func(
                    current_params_dict,  # the params dictionary
                    S_val,  # S
                    K_val,  # K
                    T_val,  # T
                )

                if verbose and _debug_counter <= 5:
                    print(f"    DEBUG: Returned price = {model_price}")

                # Convert to undiscounted price using discount factor
                undiscounted_model_price = model_price / discount_factors[i]

                # Ensure intrinsic value floor
                intrinsic_model = max(S_val - K_val, 0.0)
                undiscounted_model_price_safe = max(
                    undiscounted_model_price, intrinsic_model + 1e-12
                )

                model_iv = safe_iv_of_undiscounted_price(
                    undiscounted_model_price_safe, S_val, K_val, T_val, "c"
                )

                if verbose and _debug_counter <= 5:
                    print(
                        f"    DEBUG: Calculated IV = {model_iv:.6f}, Target IV = {target_implied_vols[i]:.6f}"  # noqa: E501
                    )

                model_implied_vols.append(model_iv)
            except Exception as e:
                if verbose:
                    print(
                        f"Warning: Invalid params {current_params_dict} caused error: {e}. Returning large error."  # noqa: E501
                    )
                return 1e6  # Penalize invalid regions heavily

        model_implied_vols = np.array(model_implied_vols)
        error = target_implied_vols - model_implied_vols

        if weights is not None:
            weighted_mse = np.average(error**2, weights=weights)
            return weighted_mse
        else:
            mse = np.mean(error**2)
            return mse

    x0 = list(initial_guess.values())
    bounds_list = [bounds[name] for name in initial_guess.keys()]

    optimizer_opts = optimizer_options or {}
    if verbose:
        optimizer_opts["disp"] = True

    if method == "differential_evolution":
        if verbose:
            print("Using Differential Evolution (global optimizer)...")
        result = differential_evolution(objective, bounds_list, **optimizer_opts)
    elif method == "L-BFGS-B":
        if verbose:
            print("Using L-BFGS-B (local optimizer)...")
        result = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds_list, options=optimizer_opts
        )
    else:
        result = minimize(
            objective, x0, method=method, bounds=bounds_list, options=optimizer_opts
        )

    if not result.success:
        print(f"Optimization status: {result.message}")
        if result.fun < 1e-3:
            print(f"Accepting suboptimal solution with MSE={result.fun:.6f}")
            recovered_params = {
                name: val for name, val in zip(list(initial_guess.keys()), result.x)
            }
            return recovered_params
        else:
            msg_lower = str(result.message).lower() if result.message else ""
            de_limits = any(
                word in msg_lower
                for word in ["iteration", "maxiter", "max evaluations"]
            )
            min_limits = any(
                word in msg_lower
                for word in ["iterations", "maxfev", "maxiter", "function evaluation"]
            )
            tolerance_reached = any(
                word in msg_lower for word in ["ftol", "xtol", "gtol", "converged"]
            )

            if de_limits or min_limits or tolerance_reached:
                print(
                    f"Optimization terminated, returning best result (MSE={result.fun:.6f})"  # noqa: E501
                )
                recovered_params = {
                    name: val for name, val in zip(list(initial_guess.keys()), result.x)
                }
                return recovered_params
            else:
                print("Returning initial guess due to optimization failure.")
                return initial_guess
    else:
        recovered_params = {
            name: val for name, val in zip(list(initial_guess.keys()), result.x)
        }
        return recovered_params
