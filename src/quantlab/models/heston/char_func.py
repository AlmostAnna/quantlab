"""
Heston characteristic function.

This module contains implementation of the log(E[exp(i*u*log(S_T))]).
"""

import numpy as np


def heston_char_func_log(u, tau, r, kappa, theta, eta, rho, v0, S0):
    """Heston characteristic function.

    Computes the log of the characteristic function for log(S_T),
    i.e., log(E[exp(i*u*log(S_T))]).
    """
    # Ensure u is treated as complex
    u_complex = np.asarray(u, dtype=complex)
    i = 1j

    # --- Heston Model Parameters for Characteristic Function ---
    # For the log-price process X_t = log(S_t),
    # dX_t = (r - 0.5*v_t) dt + sqrt(v_t) dW_t^S
    # dv_t = kappa*(theta - v_t) dt + eta*sqrt(v_t) dW_t^v
    # d<W^S, W^v>_t = rho dt

    # Coefficients for the Riccati equations solution
    # dX_t = (r - 0.5*v_t) dt + ...
    alpha = -0.5 * u_complex * (u_complex + i)
    beta = kappa - rho * eta * i * u_complex
    gamma = 0.5 * eta**2

    # Discriminant
    d_discriminant = np.sqrt(beta**2 - 4 * alpha * gamma)

    # Ensure the square root has a positive real part for convergence
    # This choice determines which branch of the solution to take
    # Use np.where for array-safe conditional logic
    d_discriminant = np.where(
        np.real(d_discriminant) < 0, -d_discriminant, d_discriminant
    )

    # --- D and C functions ---
    g = (beta - d_discriminant) / (beta + d_discriminant)

    exp_d_tau = np.exp(-d_discriminant * tau)

    # Calculate D(t, tau, u)
    denom_D = 1 - g * exp_d_tau
    # Handle potential division by zero in D
    # if np.abs(denom_D) < 1e-15:
    # L'Hôpital's rule approximation for small denominator
    #    D = (beta - d_discriminant) / (2 * gamma) * tau
    # else:
    #    D = (beta - d_discriminant) / (2 * gamma) * (1 - exp_d_tau) / denom_D
    # Use np.where for array-safe conditional logic
    D_singular = (beta - d_discriminant) / (2 * gamma) * tau
    D_regular = (beta - d_discriminant) / (2 * gamma) * (1 - exp_d_tau) / denom_D
    D = np.where(np.abs(denom_D) < 1e-15, D_singular, D_regular)

    # Calculate C(t, tau, u)
    # Part 1: Drift term
    C_drift = i * u_complex * (np.log(S0) + r * tau)

    # Part 2: Mean-reversion term
    # C_mean_rev = (kappa * theta / eta^2) * [(beta - d_discriminant)*tau
    # - 2*log((1-g*exp_d_tau)/(1-g))]
    log_fraction = (1 - g * exp_d_tau) / (1 - g)

    # Handle potential numerical issues in the logarithm
    # Use np.where for array-safe conditional logic
    log_term_singular = np.log(1e-15 + 0j)
    # Calculate standard log term
    log_term_standard = np.log(log_fraction)
    # Use standard term where condition is met, singular term otherwise
    log_term = np.where(np.abs(1 - g) < 1e-10, log_term_singular, log_term_standard)

    C_mean_rev = (kappa * theta / eta**2) * (
        (beta - d_discriminant) * tau - 2 * log_term
    )

    C = C_drift + C_mean_rev

    # Return log(phi(u))
    return C + D * v0
