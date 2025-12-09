import torch
import numpy as np


def heston_char_func_log(u: float, tau: float, r: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, S0: float):
    """
    Log of the Heston characteristic function.
    Returns log(φ(u)) = C + D*v0 + i*u*log(S0)
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u

    # Riccati coefficients
    alpha = -0.5 * (u**2 + iu)
    beta = kappa - rho * sigma * iu
    gamma = 0.5 * sigma**2

    # Discriminant
    disc = beta**2 - 4.0 * alpha * gamma
    d = np.sqrt(disc)

    # --- Enforce Re(d) >= 0 (critical for stability) ---
    if np.real(d) < 0:
        d = -d

    # Handle g = (beta - d)/(beta + d)
    denom = beta + d
    if np.abs(denom) < 1e-14:
        g = 0.0
    else:
        g = (beta - d) / denom

    # Exponential term
    exp_d_tau = np.exp(-d * tau)

    # D function
    denom_D = 1.0 - g * exp_d_tau
    if np.abs(denom_D) < 1e-14:
        # Use limit as g*exp → 1
        D = (beta - d) / sigma**2 * tau
    else:
        D = (beta - d) / sigma**2 * (1.0 - exp_d_tau) / denom_D

    # C function: careful with log argument
    log_num = 1.0 - g * exp_d_tau
    log_den = 1.0 - g
    if np.abs(log_den) < 1e-14:
        # Series expansion: log((1 - g*e^{-dτ})/(1 - g)) ≈ -g*(1 - e^{-dτ})/(1 - g) → τ*(beta - d)/2
        log_arg = tau * (beta - d) / 2.0
    else:
        log_arg = log_num / log_den
        if np.abs(log_arg) < 1e-14:
            log_arg = 1e-14
        log_arg = np.log(log_arg)

    C = r * iu * tau + (kappa * theta / sigma**2) * ((beta - d) * tau - 2.0 * log_arg)

    return C + D * v0 + iu * np.log(S0)



