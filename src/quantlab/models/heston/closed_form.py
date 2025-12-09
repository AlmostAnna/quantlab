import torch
import numpy as np

from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import vega

from quantlab.models.heston.char_func import heston_char_func_log

def heston_call_price(K: float, T: float, r: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, S0: float):
    """Robust Heston price using log-CF and adaptive quadrature."""
    logK = np.log(K)

    def integrand_P1(u):
        # u is real; we evaluate at u - i for P1
        log_phi = heston_char_func_log(u - 1j, T, r, kappa, theta, sigma, rho, v0, S0)
        # Compute real part of exp(-i*u*logK + log_phi) / (i*u)
        exponent = -1j * u * logK + log_phi
        return np.real(np.exp(exponent) / (1j * u))

    def integrand_P2(u):
        log_phi = heston_char_func_log(u, T, r, kappa, theta, sigma, rho, v0, S0)
        exponent = -1j * u * logK + log_phi
        return np.real(np.exp(exponent) / (1j * u))

    # Integrate with error handling
    try:
        I1, _ = quad(integrand_P1, 1e-10, 100, limit=200, epsabs=1e-10, epsrel=1e-8)
        I2, _ = quad(integrand_P2, 1e-10, 100, limit=200, epsabs=1e-10, epsrel=1e-8)
        P1 = 0.5 + I1 / np.pi
        P2 = 0.5 + I2 / np.pi
        return S0 * P1 - K * np.exp(-r * T) * P2
    except Exception as e:
        print(f"Integration failed: {e}")
        return np.nan


#Calibration functions
def calibration_error_vega(x, market_data, market_prices, S0, r):
    kappa, theta, sigma, rho, v0 = (
        np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), np.tanh(x[3]), np.exp(x[4])
    )
    total_error = 0.0
    for (K, T), C_mkt in zip(market_data, market_prices):
        try:
            C_model = heston_call_price(K, T, r, kappa, theta, sigma, rho, v0, S0)
            # Use a reasonable vol guess for vega
            vol_guess = 0.2
            vega = vega('c', S0, K, T, r, vol_guess)
            weight = 1.0 / (vega + 1e-8)  # avoid div by zero
            total_error += weight * (C_model - C_mkt)**2
        except Exception:
            return 1e6
    return total_error


def calibration_error(x, market_data, market_prices, S0, r):
    kappa, theta, sigma, rho, v0 = (
        np.exp(x[0]),
        np.exp(x[1]),
        np.exp(x[2]),
        np.tanh(x[3]),
        np.exp(x[4])
    )
    total_error = 0.0
    for (K, T), C_mkt in zip(market_data, market_prices):
        try:
            C_model = heston_call_price(K, T, r, kappa, theta, sigma, rho, v0, S0)
            total_error += (C_model - C_mkt)**2
        except Exception:
            return 1e6  # penalty for numerical failure
    return total_error


