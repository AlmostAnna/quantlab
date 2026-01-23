"""
Model calibration helper module.

This module contains convinience wrappers for calibration routines.
"""
from typing import Any, Callable, Dict

from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess


def make_heston_object_wrapper(
    pricer_func: Callable,
    market_state_for_calibration: MarketState,
    pricer_kwargs: Dict[str, Any] = None,
) -> Callable[[Dict[str, float], float, float, float], float]:
    """
    Create a wrapper compatible with generic calibration routines.

    This wrapper constructs the necessary MarketState, HestonProcess, and StockOption
    objects for your object-oriented pricer.

    Args:
        pricer_func: The specific pricer function (e.g., cos_price, mc_price).
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
