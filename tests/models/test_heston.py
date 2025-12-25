"""
Tests for Heston analytical pricing.

This module contains tests to ensure basic correctness of call prices in Heston model.
"""
from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.closed_form import heston_call_price
from quantlab.models.heston.model import HestonParameters, HestonProcess


def test_heston_call_price_positive():
    """Test call price range."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=0.04, kappa=1.0, theta=0.04, eta=0.3, rho=-0.5)
    process = HestonProcess(params, market_state)
    option = StockOption(strike_price=100.0, expiration_time=1.0, is_call=True)

    price = heston_call_price(option, process)

    assert price > 0.0
    assert price < 100.0  # can't be worth more than underlying


def test_heston_atm_convergence():
    """Test ATM call price convergence to max(S-K,0)."""
    # As T → 0, price → max(S-K, 0)
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=0.04, kappa=1.0, theta=0.04, eta=0.3, rho=-0.5)
    process = HestonProcess(params, market_state)
    option = StockOption(strike_price=100.0, expiration_time=1e-6, is_call=True)

    price = heston_call_price(option, process)
    # price = heston_call_price(
    #    S0=100,
    #    K=100,
    #    T=1e-6,
    #    r=0.0,  # q=0.0,
    #    kappa=1.0,
    #    theta=0.04,
    #    sigma=0.3,
    #    rho=-0.5,
    #    v0=0.04,
    # )

    assert abs(price - 0.0) < 1e-3
