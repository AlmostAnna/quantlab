"""
Market data simulators.

This module contains implementation of various data simulators.
"""
import numpy as np
from py_vollib.black.implied_volatility import (
    implied_volatility_of_undiscounted_option_price,
)

from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.closed_form import heston_call_price
from quantlab.models.heston.model import HestonParameters, HestonProcess
from quantlab.pricing.heston.cos import price as cos_price

# Vectorize py_vollib for batch IV calculation
implied_volatility_vec = np.vectorize(implied_volatility_of_undiscounted_option_price)


def generate_heston_vol_surface(
    market_state: MarketState = None,
    heston_params: HestonParameters = None,
    strikes: np.ndarray = None,
    maturities: np.ndarray = None,
    add_noise: bool = False,
    noise_level: float = 0.005,  # ±0.5% vol
    seed: int = None,
    pricing_method: str = "closed_form",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic implied volatility surface from the Heston model.

    Parameters
    ----------
    market_state : MarketState, optional
        Spot, rate, time.
        Default: S=100, r=0, t=0.
    heston_params : HestonParameters, optional
        kappa, theta, eta, rho, v0.
        Default: plausible market-like values.
    strikes : array-like, optional
        Strike grid. Default: [80, ..., 120].
    maturities : array-like, optional
        Maturity grid. Default: [0.25, 0.5, 1.0, 2.0].
    add_noise : bool
        Add Gaussian noise to implied vols.
    noise_level : float
        Std dev of noise (in vol units).
    seed : int, optional
        Random seed for noise.
    pricing_method : str
        'closed_form' for semi-analytic, 'cos' for COS method.

    Returns
    -------
    strikes_flat : np.ndarray
        1D array of strikes.
    maturities_flat : np.ndarray
        1D array of maturities (T - t).
    implied_vols_flat : np.ndarray
        1D array of implied vols.
    """
    if market_state is None:
        market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)

    if heston_params is None:
        heston_params = HestonParameters(
            v0=0.04, kappa=1.5, theta=0.04, eta=0.3, rho=-0.5
        )

    if strikes is None:
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])

    if maturities is None:
        maturities = np.array([0.25, 0.5, 1.0, 2.0])

    if seed is not None:
        np.random.seed(seed)

    # --- Vectorized Surface Generation ---
    strikes_grid, maturities_grid = np.meshgrid(strikes, maturities, indexing="ij")
    # (nK, nT), (nK, nT)

    # Flatten grids for iteration
    strikes_flat = strikes_grid.flatten()
    maturities_flat = maturities_grid.flatten()  # This is T (absolute time)
    times_to_expiry_flat = maturities_flat - market_state.time  # This is T - t

    # --- Price Options ---
    prices_flat = []
    for K, T_abs in zip(strikes_flat, maturities_flat):
        option = StockOption(strike_price=K, expiration_time=T_abs, is_call=True)

        # Create process for this specific time (or assume it's stateless for pricing)
        process = HestonProcess(model_params=heston_params, market_state=market_state)

        if pricing_method == "closed_form":
            price = heston_call_price(option, process)
        elif pricing_method == "cos":
            price = cos_price(option, process)
        else:
            raise ValueError(f"Unknown pricing method: {pricing_method}")

        prices_flat.append(price)

    prices_flat = np.array(prices_flat)

    # --- Convert Prices to IV using py_vollib ---
    # 1. Calculate forward prices and discount factors
    forward_prices = market_state.stock_price * np.exp(
        market_state.interest_rate * times_to_expiry_flat
    )
    discount_factors = np.exp(-market_state.interest_rate * times_to_expiry_flat)

    # 2. Convert discounted prices to undiscounted prices
    undiscounted_prices = prices_flat / discount_factors

    # 3. Use py_vollib to get IV
    # py_vollib expects (undiscounted_price, forward, strike, time, flag)
    iv_flat = implied_volatility_vec(
        undiscounted_prices,
        forward_prices,
        strikes_flat,
        times_to_expiry_flat,  # T - t
        "c",  # call
    )

    # Apply noise if requested
    if add_noise:
        noise = np.random.normal(0, noise_level, size=iv_flat.shape)
        iv_flat = np.clip(iv_flat + noise, 0.01, 2.0)

    # Return flattened arrays
    # Note: We return times_to_expiry_flat, which is T - t
    return strikes_flat, times_to_expiry_flat, iv_flat


def generate_heston_bid_ask_spreads(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
    bid_ask_half_width: float = 0.0025,  # ±25 bps
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic bid/ask spreads around a mid implied volatility surface.

    Args:
        strikes : array-like
            Strike grid.
        maturities : array-like
            Maturity grid.
        implied_vols: array-like Implied Volitilities.
        bid_ask_half_width: 1/2 of the bid/ask spread.

    Returns:
        A tuple of bid IVs and ask IVs.
    """
    iv_bid = np.maximum(implied_vols - bid_ask_half_width, 0.01)
    iv_ask = implied_vols + bid_ask_half_width
    return iv_bid, iv_ask
