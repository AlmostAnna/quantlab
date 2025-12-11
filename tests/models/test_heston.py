# tests/models/test_heston.py
"""
Tests for Heston analytical pricing.

This module contains tests to ensure basic correctness of call prices in Heston model.
"""

from quantlab.models.heston.closed_form import heston_call_price


def test_heston_call_price_positive():
    """Test call price range."""
    price = heston_call_price(
        S0=100,
        K=100,
        T=1.0,
        r=0.0,  # q=0.0,
        kappa=1.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
        v0=0.04,
    )
    assert price > 0.0
    assert price < 100.0  # can't be worth more than underlying


def test_heston_atm_convergence():
    """Test ATM call price convergence to max(S-K,0)."""
    # As T → 0, price → max(S-K, 0)
    price = heston_call_price(
        S0=100,
        K=100,
        T=1e-6,
        r=0.0,  # q=0.0,
        kappa=1.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
        v0=0.04,
    )
    assert abs(price - 0.0) < 1e-3
