# This module uses QuantLib (https://www.quantlib.org/),
# licensed under the BSD 3-Clause License. See NOTICES for details.

"""
Tests for comparing Heston analytical price implementation with QuantLib.

This module contains tests to ensure prices are consistent with baseline.
"""

import pytest
from quantlib_helper import heston_call_price_QL, prices_close

from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.closed_form import heston_call_price
from quantlab.models.heston.model import HestonParameters, HestonProcess


@pytest.mark.parametrize(
    "kappa,theta,eta,rho,v0,K,T",
    [
        (1.0, 0.04, 0.3, -0.5, 0.04, 100, 1.0),  # Heston 1993
        (2.0, 0.05, 0.2, -0.7, 0.06, 90, 0.5),  # OTM
        (0.5, 0.03, 0.4, -0.3, 0.02, 110, 2.0),  # ITM, long-dated
    ],
)
def test_heston_vs_quantlib_parametrized(kappa, theta, eta, rho, v0, K, T):
    """Test closeness of prices."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=v0, kappa=kappa, theta=theta, eta=eta, rho=rho)
    process = HestonProcess(params, market_state)
    option = StockOption(strike_price=K, expiration_time=T, is_call=True)

    price = heston_call_price(option, process)

    price_ql = heston_call_price_QL(
        K=K,
        T=T,
        r=0.0,
        q=0.0,
        kappa=kappa,
        theta=theta,
        eta=eta,
        rho=rho,
        v0=v0,
        S0=100,
    )

    assert prices_close(price, price_ql)
