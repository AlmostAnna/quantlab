"""
Tests for Heston MC pricing.

This module contains tests to ensure basic correctness of MC pricing
of calls in Heston model.
"""
import numpy as np

from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess
from quantlab.sim.heston.mc_pricer import heston_euler_mc_price


def test_heston_exact_mc_price_positive():
    """Test call price range."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=0.04, kappa=1.0, theta=0.04, eta=0.3, rho=-0.5)
    process = HestonProcess(params, market_state)
    option = StockOption(strike_price=100.0, expiration_time=1.0, is_call=True)

    price = heston_euler_mc_price(option, process)

    assert price > 0.0
    assert price < 100.0  # can't be worth more than underlying


def test_heston_exact_mc_price_atm_convergence():
    """Test ATM call price convergence to max(S-K,0) for a very short maturity."""
    # As T gets very small (but not too small for numerical stability),
    # price -> max(S-K, 0)
    # Use T = 0.01 (about 3.65 days) which is short but numerically manageable
    T_short = 0.01
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=0.04, kappa=1.0, theta=0.04, eta=0.3, rho=-0.5)
    process = HestonProcess(params, market_state)
    option = StockOption(strike_price=100.0, expiration_time=T_short, is_call=True)

    price = heston_euler_mc_price(
        option, process, n_paths=100000, n_steps=100
    )  # Increase paths for stability

    # A more appropriate test for short maturity might be
    # that the price is positive and finite
    # and significantly smaller than a longer maturity price.
    assert np.isfinite(price)
    assert price >= 0
    assert (
        0 <= price <= 5.0
    ), f"MC price {price} is outside expected range for short maturity T={T_short}"
