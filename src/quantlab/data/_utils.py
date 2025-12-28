"""
Data generation helper module.

This module contains convinience wrappers for py_vollib.
"""
import numpy as np
from py_vollib.black.implied_volatility import (
    implied_volatility_of_undiscounted_option_price,
)


def _price_to_iv_py_vollib(price, S, K, T, r, q=0.0):
    """
    Convert option price to implied volatility using py_vollib.

    Parameters
    ----------
    price : float
        Discounted option price from Heston model.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield (default 0).

    Returns
    -------
    float
        Implied volatility, or np.nan if inversion fails.
    """
    if price <= 0:
        return np.nan

    # py_vollib expects undiscounted price: multiply by exp(r*T)
    undiscounted_price = price * np.exp(r * T)

    try:
        # py_vollib uses forward price: F = S * exp((r-q)*T)
        F = S * np.exp((r - q) * T)
        iv = implied_volatility_of_undiscounted_option_price(
            undiscounted_price, F, K, T, "c"  # 'c' for call
        )
        return iv
    except (ValueError, OverflowError):
        # Inversion failed (e.g., price too small, F/K invalid)
        return np.nan
