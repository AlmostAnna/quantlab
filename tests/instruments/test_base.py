"""Tests for basic European option."""
import numpy as np
import pytest

from quantlab.instruments.base import StockOption


def test_single():
    """Test single option payoff."""
    opt_single = StockOption(strike_price=100.0, expiration_time=1.0, is_call=True)
    payoff_single = opt_single.payoff(stock_price=105.0)

    assert payoff_single == pytest.approx(5.0)


def test_multiple():
    """Test vectorized payoff."""
    strikes = np.array([100.0, 105.0, 95.0])
    expiries = np.array([1.0, 0.5, 2.0])
    is_calls = np.array([True, False, True])  # Mixed calls and puts
    opt_batch = StockOption(
        strike_price=strikes, expiration_time=expiries, is_call=is_calls
    )
    stock_prices_at_expiry = np.array([102.0, 103.0, 90.0])
    payoffs_batch = opt_batch.payoff(stock_price=stock_prices_at_expiry)

    # Result: [max(102-100,0), max(105-103,0), max(90-95,0)] = [2.0, 2.0, 0.0]
    assert payoffs_batch == pytest.approx([2.0, 2.0, 0.0])
