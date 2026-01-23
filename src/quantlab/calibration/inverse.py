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
                S_val = forward if np.isscalar(forward) else forward[i]
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
                model_iv = implied_volatility_of_undiscounted_option_price(
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
        mse = np.mean(error**2)
        if verbose and _debug_counter <= 5:
            print(f"  DEBUG: Calculated MSE = {mse:.8f}\n")
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
        print(f"Optimization failed: {result.message}")
        return initial_guess

    recovered_params = {
        name: val for name, val in zip(list(initial_guess.keys()), result.x)
    }
    return recovered_params
