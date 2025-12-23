"""
Heston model based MC pricers.

This module contains implementation of MC pricers for Heston model.
"""
import numpy as np
from scipy.stats import ncx2

from quantlab.instruments.base import StockOption
from quantlab.models.heston.model import HestonProcess


def heston_exact_mc_price(
    option: StockOption,
    process: HestonProcess,
    n_paths: int = 50000,
    n_steps: int = 200,
    seed: int = 42,
):
    """
    Price for the StockOption option using MC.

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

    # --- Precompute constants for the EXACT CIR step ---
    exp_kappa_dt = np.exp(-kappa * dt)
    c1 = eta**2 * (1 - exp_kappa_dt) / (4 * kappa)
    c2 = 4 * kappa * exp_kappa_dt / (eta**2 * (1 - exp_kappa_dt))
    d = 4 * kappa * theta / eta**2  # degrees of freedom

    # Initialize paths
    v = np.full(n_paths, v0)
    S = np.full(n_paths, S0)

    # --- Simulation Loop ---
    for i in range(n_steps):
        Z1 = np.random.randn(n_paths)  # Independent BM for S
        Z2 = np.random.randn(n_paths)  # Independent BM for v

        # --- 1. Asset step (Log-Euler) using current variance v ---
        S *= np.exp(
            (r - 0.5 * v) * dt
            + np.sqrt(v * dt) * (rho * Z2 + np.sqrt(1 - rho**2) * Z1)
        )

        # --- 2. Variance step: EXACT simulation using precomputed constants ---
        # This is the optimized part that uses precomputed c1, c2, d
        lam = c2 * v  # non-centrality parameter (vectorized)
        chi2 = ncx2.rvs(d, lam, size=n_paths)  # Sample from ncx2 for all paths
        v = c1 * chi2  # update variance for all paths

    # Calculate payoff using the option object
    payoff = option.payoff(S)
    return np.exp(-r * T) * np.mean(payoff)


def heston_exact_mc_price_with_paths(
    S0, K, T, r, kappa, theta, sigma, rho, v0, Z1_base, Z2_base, n_steps
):
    """
    Price using pre-generated random paths (for CRN).

    Z1_base, Z2_base: shape (n_paths, n_steps)
    """
    n_paths = Z1_base.shape[0]
    dt = T / n_steps
    exp_kappa_dt = np.exp(-kappa * dt)
    c1 = sigma**2 * (1 - exp_kappa_dt) / (4 * kappa)
    c2 = 4 * kappa * exp_kappa_dt / (sigma**2 * (1 - exp_kappa_dt))
    d = 4 * kappa * theta / sigma**2

    v = np.full(n_paths, v0)
    S = np.full(n_paths, S0)

    for i in range(n_steps):
        Z1 = Z1_base[:, i]
        Z2 = Z2_base[:, i]

        # Asset step (using current v)
        S *= np.exp(
            (r - 0.5 * v) * dt
            + np.sqrt(v * dt) * (rho * Z2 + np.sqrt(1 - rho**2) * Z1)
        )

        # Variance step
        lam = c2 * v
        chi2 = ncx2.rvs(d, lam, size=n_paths)
        v = c1 * chi2

    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * np.mean(payoff)
