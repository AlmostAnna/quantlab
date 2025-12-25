"""
Heston analytical pricing.

This module contains analytic pricing function implementation.
"""

import numpy as np
from scipy.integrate import quad

from quantlab.instruments.base import StockOption
from quantlab.models.heston.char_func import heston_char_func_log
from quantlab.models.heston.model import HestonProcess


def heston_call_price(
    option: StockOption,
    process: HestonProcess,
):
    """
    Robust Heston price using log-CF and adaptive quadrature.

     Args:
        option: Stock option to price.
        process: Underlying dynamics.

    Returns:
        A price for the StockOption option.
    """
    # Extract parameters
    market_state = process.market_state

    S0 = market_state.stock_price
    T = option.expiration_time
    K = option.strike_price
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

    logK = np.log(K)

    def integrand_P1(u):
        # P1 integrand using phi2(u-i) with proper scaling
        log_phi_Q = heston_char_func_log(u - 1j, T, r, kappa, theta, eta, rho, v0, S0)
        # phi_S(u) = phi_Q(u-i) * exp(r*T) / S0
        # integrand = Re[ exp(-i*u*logK) * phi_Q(u-i) * exp(r*T) / (i*u*S0) ]
        exp_factor = np.exp(-1j * u * logK + r * T)
        phi_Q_shifted = np.exp(log_phi_Q)
        complex_integrand = (exp_factor * phi_Q_shifted) / (1j * u * S0)
        return np.real(complex_integrand)

    def integrand_P2(u):
        # P2 uses the standard risk-neutral CF
        log_phi = heston_char_func_log(u, T, r, kappa, theta, eta, rho, v0, S0)
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
