"""
Heston analytical pricing and model calibration functions.

This module contains analytic pricing function implementation
and two model calibration objectives: equal and vega-weighted.
"""

import numpy as np
from py_vollib.black_scholes.greeks.analytical import vega
from scipy.integrate import quad

from quantlab.models.heston.char_func import heston_char_func_log


def heston_call_price(
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    S0: float,
):
    """Robust Heston price using log-CF and adaptive quadrature."""
    logK = np.log(K)

    def integrand_P1(u):
        # P1 integrand using phi2(u-i) with proper scaling
        log_phi_Q = heston_char_func_log(u - 1j, T, r, kappa, theta, sigma, rho, v0, S0)
        # phi_S(u) = phi_Q(u-i) * exp(r*T) / S0
        # integrand = Re[ exp(-i*u*logK) * phi_Q(u-i) * exp(r*T) / (i*u*S0) ]
        exp_factor = np.exp(-1j * u * logK + r * T)
        phi_Q_shifted = np.exp(log_phi_Q)
        complex_integrand = (exp_factor * phi_Q_shifted) / (1j * u * S0)
        return np.real(complex_integrand)

    def integrand_P2(u):
        # P2 uses the standard risk-neutral CF
        log_phi = heston_char_func_log(u, T, r, kappa, theta, sigma, rho, v0, S0)
        exp_factor = np.exp(-1j * u * logK)
        phi_val = np.exp(log_phi)
        complex_integrand = (exp_factor * phi_val) / (1j * u)
        return np.real(complex_integrand)

    # Integrate with error handling, starting from a small positive number to avoid u=0
    try:
        # Use slightly larger limits and looser tolerances initially
        I1, _ = quad(integrand_P1, 1e-10, 50, limit=500, epsabs=1e-10, epsrel=1e-8)
        I2, _ = quad(integrand_P2, 1e-10, 50, limit=500, epsabs=1e-10, epsrel=1e-8)
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
        price = S0 * P1 - K * np.exp(-r * T) * P2
        return price
    except Exception as e:
        print(f"Integration failed: {e}")
        return np.nan


# Calibration functions
def calibration_error_vega(x, market_data, market_prices, S0, r):
    """Calibration objective function with vega weights."""
    kappa, theta, sigma, rho, v0 = (
        np.exp(x[0]),
        np.exp(x[1]),
        np.exp(x[2]),
        np.tanh(x[3]),
        np.exp(x[4]),
    )
    total_error = 0.0
    for (K, T), C_mkt in zip(market_data, market_prices):
        try:
            C_model = heston_call_price(K, T, r, kappa, theta, sigma, rho, v0, S0)
            # Use a reasonable vol guess for vega
            vol_guess = 0.2
            vga = vega("c", S0, K, T, r, vol_guess)
            weight = 1.0 / (vga + 1e-8)  # avoid div by zero
            total_error += weight * (C_model - C_mkt) ** 2
        except Exception:
            return 1e6
    return total_error


def calibration_error(x, market_data, market_prices, S0, r):
    """Calibration objective function with equal weights."""
    kappa, theta, sigma, rho, v0 = (
        np.exp(x[0]),
        np.exp(x[1]),
        np.exp(x[2]),
        np.tanh(x[3]),
        np.exp(x[4]),
    )
    total_error = 0.0
    for (K, T), C_mkt in zip(market_data, market_prices):
        try:
            C_model = heston_call_price(K, T, r, kappa, theta, sigma, rho, v0, S0)
            total_error += (C_model - C_mkt) ** 2
        except Exception:
            return 1e6  # penalty for numerical failure
    return total_error
