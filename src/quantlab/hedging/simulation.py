"""
Hedging simulation module.

This module contains functions for simulating delta hedging strategies
under various models.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from quantlab.hedging.greeks import heston_delta_bump_revalue
from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonProcess
from quantlab.pricing.heston.cos import price as cos_price


def build_delta_interpolator(
    option: StockOption,
    process: HestonProcess,
    S_grid: np.ndarray,
    T_grid: np.ndarray,
    n_paths: int = 10000,
    n_steps: int = 100,
    bump_size: float = 0.01,
    seed: int = 42,
) -> RegularGridInterpolator:
    """
    Precompute delta surface for interpolation during hedging simulation.

    Args:
        option: Option to hedge
        process: Heston process parameters
        S_grid: Grid of stock prices
        T_grid: Grid of times to maturity
        n_paths: Number of paths for MC delta calculation
        n_steps: Number of time steps for MC delta calculation
        bump_size: Bump size for finite difference
        seed: Random seed for reproducibility

    Returns:
        Interpolator function for delta lookup
    """
    deltas = np.zeros((len(T_grid), len(S_grid)))

    for i, T in enumerate(T_grid):
        # Create temporary option with adjusted maturity
        temp_option = StockOption(
            strike_price=option.strike_price, expiration_time=T, is_call=option.is_call
        )

        for j, S in enumerate(S_grid):
            # Create temporary process with current stock price
            temp_market_state = MarketState(
                stock_price=S,
                interest_rate=process.market_state.interest_rate,
                time=process.market_state.time,
            )
            temp_process = HestonProcess(process.model_params, temp_market_state)

            delta, _, _, _ = heston_delta_bump_revalue(
                temp_option,
                temp_process,
                n_paths=n_paths,
                n_steps=n_steps,
                bump_size=bump_size,
                seed=seed,
            )
            deltas[i, j] = delta

    return RegularGridInterpolator(
        (T_grid, S_grid), deltas, bounds_error=False, fill_value=None
    )


def simulate_heston_hedging_path(
    option: StockOption,
    process: HestonProcess,
    initial_price: float,
    delta_interpolator: RegularGridInterpolator,
    n_steps: int,
    seed: int = 42,
) -> tuple[float, list[float], list[float], list[float]]:
    """
    Simulate a single hedging path under Heston model.

    Args:
        option: Option to hedge
        process: Heston process parameters
        initial_price: Initial stock price
        delta_interpolator: Pre-computed delta interpolator
        n_steps: Number of rebalancing steps
        seed: Random seed

    Returns:
        (hedging_pnl, stock_path, variance_path, delta_path)
    """
    np.random.seed(seed)

    # Extract parameters
    T = option.expiration_time
    r = process.market_state.interest_rate
    params = process.model_params
    kappa, theta, eta, rho, v0 = (
        params.kappa,
        params.theta,
        params.eta,
        params.rho,
        params.v0,
    )

    dt = T / n_steps

    # Initialize
    S = initial_price
    v = v0
    portfolio_value = 0.0  # cash account value

    # Create temporary process with current stock price for initial pricing
    initial_market_state = MarketState(
        stock_price=S,
        interest_rate=process.market_state.interest_rate,
        time=process.market_state.time,
    )
    initial_process = HestonProcess(process.model_params, initial_market_state)

    # Calculate initial option price
    initial_option_price = cos_price(option, initial_process)

    # Get initial delta from interpolator
    tau = T  # Initial time to maturity
    initial_delta = delta_interpolator((tau, S))[()]

    # Initial hedging: sell option, buy delta shares
    portfolio_value = initial_option_price - initial_delta * S

    stock_path = [initial_price]
    variance_path = [v0]
    delta_path = [initial_delta]

    for step in range(n_steps):
        t_step = (step + 1) * dt
        tau = T - t_step  # Time to maturity decreases

        # Generate correlated normals
        Z1 = np.random.randn()
        Z2 = np.random.randn()
        Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlated normal for variance

        # Stock price step (Log-Euler)
        S_new = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * Z_v)

        # Variance step (Euler, similar to heston_euler_mc_price)
        dv = kappa * (theta - v) * dt + eta * np.sqrt(max(v, 0.0)) * np.sqrt(dt) * Z2
        v_new = v + dv
        v_new = max(v_new, 0.0)  # Ensure non-negative variance

        # Get new delta for the next period (after price moves to S_new)
        delta_new = delta_interpolator((tau, S_new))[()]

        # Rebalance: adjust delta position
        d_delta = delta_new - initial_delta
        portfolio_value -= d_delta * S_new  # cost of buying/selling shares
        initial_delta = delta_new

        # Add the new delta to the path (for the next time period)
        delta_path.append(initial_delta)

        # Update paths
        stock_path.append(S_new)
        variance_path.append(v_new)

        S, v = S_new, v_new

    # Final payoff
    final_payoff = option.payoff(S)

    # Final hedge value: liquidate final delta position
    final_hedge_value = initial_delta * S
    final_portfolio = portfolio_value + final_hedge_value

    # Hedging P&L calculation
    hedging_pnl = initial_option_price - (final_payoff - final_portfolio)

    return hedging_pnl, stock_path, variance_path, delta_path


def simulate_multiple_hedging_paths(
    option: StockOption,
    process: HestonProcess,
    initial_price: float,
    delta_interpolator: RegularGridInterpolator,
    n_paths: int,
    n_steps: int,
    seed_start: int = 1000,
) -> np.ndarray:
    """
    Simulate multiple hedging paths and return P&L distribution.

    Args:
        option: Option to hedge
        process: Heston process parameters
        initial_price: Initial stock price
        delta_interpolator: Pre-computed delta interpolator
        n_paths: Number of simulation paths
        n_steps: Number of rebalancing steps per path
        seed_start: Starting seed (will increment for each path)

    Returns:
        Array of hedging P&Ls for each path
    """
    pnl_list = []

    for i in range(n_paths):
        pnl, _, _, _ = simulate_heston_hedging_path(
            option=option,
            process=process,
            initial_price=initial_price,
            delta_interpolator=delta_interpolator,
            n_steps=n_steps,
            seed=seed_start + i,
        )
        pnl_list.append(pnl)

    return np.array(pnl_list)
