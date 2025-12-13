"""
Heston model calibration functions.

This module contains two model calibration objectives: equal and vega-weighted.
"""

import numpy as np
from py_vollib.black_scholes.greeks.analytical import vega

from quantlab.models.heston.closed_form import heston_call_price


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
