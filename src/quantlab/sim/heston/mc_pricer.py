"""
Heston model based MC pricers.

This module contains implementation of MC pricers for Heston model.
"""
import numpy as np

from quantlab.instruments.base import StockOption
from quantlab.models.heston.model import HestonProcess


def heston_euler_mc_price(
    option: StockOption,
    process: HestonProcess,
    n_paths: int = 500000,  # Use more paths to compensate for discretization error
    n_steps: int = 1000,  # Use more steps to reduce discretization error
    seed: int = 42,
):
    """
    Price for the StockOption option using MC.

    Not recommended for ITM, long-dated options.

    Args:
        option: Stock option to price.
        process: Underlying dynamics.
        n_paths: The number of paths to generate.
        n_steps: The number of steps in each path.
        seed: The seed for np.random.

    Returns:
        A price for the StockOption option.
    """
    # Extract parameters
    market_state = process.market_state

    S0 = market_state.stock_price
    T = option.expiration_time
    r = market_state.interest_rate

    # Extract model parameters
    params = process.model_params  # HestonParameters
    kappa, theta, eta, rho, v0 = (
        params.kappa,
        params.theta,
        params.eta,
        params.rho,
        params.v0,
    )

    np.random.seed(seed)
    dt = T / n_steps

    # Initialize paths
    v = np.full(n_paths, v0)
    S = np.full(n_paths, S0)

    for i in range(n_steps):
        Z1 = np.random.randn(n_paths)  # Independent BM for S
        Z2 = np.random.randn(n_paths)  # Independent BM for v

        # --- Stock Price Step (Log-Euler) ---
        # Uses CURRENT variance v
        S *= np.exp(
            (r - 0.5 * v) * dt
            + np.sqrt(v * dt) * (rho * Z2 + np.sqrt(1 - rho**2) * Z1)
        )

        # --- Variance Step (Euler) ---
        # Uses CURRENT variance v
        dv = (
            kappa * (theta - v) * dt
            + eta * np.sqrt(np.maximum(v, 0.0)) * np.sqrt(dt) * Z2
        )
        v = v + dv
        # Ensure variance stays non-negative (standard fix for Euler)
        v = np.maximum(v, 0.0)

    payoff = option.payoff(S)
    return np.exp(-r * T) * np.mean(payoff)


def heston_euler_mc_price_with_paths(
    option: StockOption, process: HestonProcess, Z1_base, Z2_base, n_steps: int
):
    """
    Price using pre-generated random paths (for CRN).

    Args:
        option: Stock option to price.
        process: Underlying dynamics.
        Z1_base: shape (n_paths, n_steps)
        Z2_base: shape (n_paths, n_steps)
        n_steps: The number of steps in each path.

    Returns:
        A price for the StockOption option.
    """
    # Extract parameters
    market_state = process.market_state

    S0 = market_state.stock_price
    T = option.expiration_time
    r = market_state.interest_rate

    # Extract model parameters
    params = process.model_params  # HestonParameters
    kappa, theta, eta, rho, v0 = (
        params.kappa,
        params.theta,
        params.eta,
        params.rho,
        params.v0,
    )

    n_paths = Z1_base.shape[0]
    dt = T / n_steps

    # Initialize paths
    v = np.full(n_paths, v0)
    S = np.full(n_paths, S0)

    for i in range(n_steps):
        Z1 = Z1_base[:, i]  # Use pregenerated paths
        Z2 = Z2_base[:, i]  # Use pregenerated paths

        # --- Stock Price Step (Log-Euler) ---
        # Uses CURRENT variance v
        S *= np.exp(
            (r - 0.5 * v) * dt
            + np.sqrt(v * dt) * (rho * Z2 + np.sqrt(1 - rho**2) * Z1)
        )

        # --- Variance Step (Euler) ---
        # Uses CURRENT variance v
        dv = (
            kappa * (theta - v) * dt
            + eta * np.sqrt(np.maximum(v, 0.0)) * np.sqrt(dt) * Z2
        )
        v = v + dv
        # Ensure variance stays non-negative (standard fix for Euler)
        v = np.maximum(v, 0.0)

    payoff = option.payoff(S)
    return np.exp(-r * T) * np.mean(payoff)
