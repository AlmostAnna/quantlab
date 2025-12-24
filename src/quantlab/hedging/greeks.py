"""
Greeks.

This module contains implementations of greeks calculations in different models.
"""
import numpy as np

from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonProcess
from quantlab.sim.heston.mc_pricer import heston_euler_mc_price_with_paths


def heston_delta_bump_revalue(
    option: StockOption,
    process: HestonProcess,
    n_paths: int = 100000,
    n_steps: int = 500,
    bump_size: float = 0.01,
    seed: int = 42,
):
    """
    Compute delta using bump-and-revalue with common random numbers.

    Args:
        option: Stock option to price.
        process: Underlying dynamics.
        n_paths: The number of paths to generate.
        n_steps: The number of steps in each path.
        bump_size: size of the bump, e.g., 0.01 for 1% bump for S0=1
        seed: The seed for np.random.

    Returns:
        A price for the StockOption option.
    """
    np.random.seed(seed)

    # Pre-generate ALL random numbers (CRN)
    Z1_base = np.random.randn(n_paths, n_steps)
    Z2_base = np.random.randn(n_paths, n_steps)

    # Base price
    base_process = process
    base_market_state = base_process.market_state
    S0 = base_market_state.stock_price

    C0 = heston_euler_mc_price_with_paths(option, process, Z1_base, Z2_base, n_steps)

    # Up bump
    up_market_state = MarketState(
        stock_price=S0 + bump_size,
        interest_rate=base_market_state.interest_rate,
        time=base_market_state.time,
    )
    up_process = HestonProcess(
        model_params=base_process.model_params, market_state=up_market_state
    )

    C_up = heston_euler_mc_price_with_paths(
        option, up_process, Z1_base, Z2_base, n_steps
    )

    # Down bump
    down_market_state = MarketState(
        stock_price=S0 - bump_size,
        interest_rate=base_market_state.interest_rate,
        time=base_market_state.time,
    )
    down_process = HestonProcess(
        model_params=base_process.model_params, market_state=down_market_state
    )

    C_down = heston_euler_mc_price_with_paths(
        option, down_process, Z1_base, Z2_base, n_steps
    )

    delta = (C_up - C_down) / (2 * bump_size)
    return delta, C0, C_up, C_down
