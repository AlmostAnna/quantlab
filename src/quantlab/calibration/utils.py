"""
Model calibration helper module.

This module contains helper functions for calibration pipelines.
"""
from typing import Any, Callable, Dict

import numpy as np
from py_vollib.black import black
from py_vollib.black.greeks.analytical import vega as bs_vega
from py_vollib.black.implied_volatility import (
    implied_volatility_of_undiscounted_option_price,
)

from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess


def calculate_vega_weights(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    forwards: np.ndarray,
    interest_rates: np.ndarray,
    min_weight: float = 1e-6,
) -> np.ndarray:
    """
    Calculate Black-Scholes vega weights for options in a volatility surface.

    Args:
        strikes: Array of strike prices
        maturities: Array of times to maturity
        implied_vols: Array of implied volatilities
        forwards: Array of forward prices
        interest_rates: Array of interest rates (for each maturity)
        min_weight: Minimum weight to avoid numerical issues

    Returns:
        Array of vega weights
    """
    vega_weights = []
    for F, K, T, iv, r in zip(
        forwards, strikes, maturities, implied_vols, interest_rates
    ):
        weight = bs_vega("c", F, K, T, r, iv)
        vega_weights.append(max(weight, min_weight))

    return np.array(vega_weights)


def make_heston_object_wrapper(
    pricer_func: Callable,
    market_state_for_calibration: MarketState,
    pricer_kwargs: Dict[str, Any] = None,
) -> Callable[[Dict[str, float], float, float, float], float]:
    """
    Create a wrapper compatible with generic calibration routines.

    This wrapper constructs the necessary MarketState, HestonProcess, and StockOption
    objects for the object-oriented pricer.

    Args:
        pricer_func: The specific pricer function.
                     Signature should be
                     (option: StockOption, process: HestonProcess, **kwargs) -> price
        market_state_for_calibration: The MarketState object representing the snapshot
                                    for calibration. This should contain the
                                    stock_price, interest_rate, time used
                                    for pricing.
        pricer_kwargs: Optional dict of kwargs to pass to the pricer function.

    Returns:
        A function with signature
        (params_dict: Dict, S: float, K: float, T: float) -> price (float)
    """

    def wrapper(
        heston_params_dict: Dict[str, float], S: float, K: float, T: float
    ) -> float:
        # 1. Create HestonParameters from the dictionary provided by the optimizer
        heston_params_obj = HestonParameters(**heston_params_dict)

        # 2. Create HestonProcess using the fixed MarketState and
        # the current HestonParameters
        heston_process = HestonProcess(heston_params_obj, market_state_for_calibration)

        # 3. Create StockOption for the specific S, K, T of this observation
        option = StockOption(strike_price=K, expiration_time=T, is_call=True)

        # 4. Call the pricer function with the constructed objects
        try:
            result = pricer_func(option, heston_process, **(pricer_kwargs or {}))
            # Ensure the result is a scalar float
            return float(result.item() if hasattr(result, "item") else result)
        except Exception as e:
            print(
                f"Pricing error for params {heston_params_dict}, S={S}, K={K}, T={T}: {e}"  # noqa: E501
            )
            raise  # Re-raise to trigger error handling in the objective function

    return wrapper


def safe_implied_volatility_of_undiscounted_price(
    undiscounted_price, F, K, T, flag, min_iv=1e-6, max_iv=5.0, tolerance=1e-8
):
    """Safely calculate implied volatility with fallbacks."""
    try:
        # Handle edge case: time to maturity is effectively zero
        if T <= tolerance:
            # When T is zero, we can't calculate implied volatility meaningfully
            # Use a reasonable fallback based on moneyness
            if flag == "c":
                intrinsic = max(0, F - K)
            else:  # flag == "p"
                intrinsic = max(0, K - F)

            if undiscounted_price <= intrinsic + tolerance:
                # Price is at or below intrinsic value
                return min_iv
            else:
                # Price exceeds intrinsic, assign minimum vol
                return min_iv

        iv = implied_volatility_of_undiscounted_option_price(
            undiscounted_price, F, K, T, flag
        )
        if np.isnan(iv) or iv < min_iv or iv > max_iv:
            raise ValueError(f"Invalid IV: {iv}")
        return iv
    except (ValueError, TypeError, RuntimeError, OverflowError, ZeroDivisionError):
        try:
            # Calculate intrinsic value
            if flag == "c":
                intrinsic = max(0, F - K)
            else:  # flag == "p"
                intrinsic = max(0, K - F)

            # If price is too close to intrinsic, return minimum vol
            if abs(undiscounted_price - intrinsic) < tolerance:
                return min_iv

            # Bisection search for IV
            low, high = min_iv, max_iv
            for _ in range(50):  # Max iterations
                mid = (low + high) / 2
                # Use Black model for bisection
                try:
                    price_at_mid = black(
                        flag, F, K, T, mid, 0.0
                    )  # r=q=0 for undiscounted
                except ZeroDivisionError:
                    # If Black model fails with T=0, use minimum vol
                    if T <= tolerance:
                        return min_iv
                    raise

                if abs(price_at_mid - undiscounted_price) < tolerance:
                    return mid
                elif price_at_mid < undiscounted_price:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2
        except (ValueError, TypeError, RuntimeError, OverflowError, ZeroDivisionError):
            return min_iv
