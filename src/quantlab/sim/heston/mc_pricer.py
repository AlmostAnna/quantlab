import torch
import numpy as np


def heston_exact_mc_price(S0, K, T, r, kappa, theta, sigma, rho, v0, n_paths=50000, n_steps=200, seed=42):
    np.random.seed(seed)
    dt = T / n_steps
    
    # Precompute constants for exact CIR
    exp_kappa_dt = np.exp(-kappa * dt)
    c1 = sigma**2 * (1 - exp_kappa_dt) / (4 * kappa)
    c2 = 4 * kappa * exp_kappa_dt / (sigma**2 * (1 - exp_kappa_dt))
    d = 4 * kappa * theta / sigma**2  # degrees of freedom (should be > 2)

    # Initialize
    v = np.full(n_paths, v0)
    S = np.full(n_paths, S0)
    
    for i in range(n_steps):
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        
        # --- 1. Asset step using CURRENT variance v ---
        S *= np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * (rho * Z2 + np.sqrt(1 - rho**2) * Z1))
        
        # --- 2. Variance step: simulate v_{t+dt} from v_t ---
        lam = c2 * v  # non-centrality parameter
        chi2 = ncx2.rvs(d, lam, size=n_paths)
        v = c1 * chi2  # update to next variance
    
    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * np.mean(payoff)

def heston_exact_mc_price_with_paths(S0, K, T, r, kappa, theta, sigma, rho, v0,
                                    Z1_base, Z2_base, n_steps):
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
        S *= np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * (rho * Z2 + np.sqrt(1 - rho**2) * Z1))

        # Variance step
        lam = c2 * v
        chi2 = ncx2.rvs(d, lam, size=n_paths)
        v = c1 * chi2

    payoff = np.maximum(S - K, 0.0)
    return np.exp(-r * T) * np.mean(payoff)

