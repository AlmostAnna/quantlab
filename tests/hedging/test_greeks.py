# This module uses QuantLib (https://www.quantlib.org/),
# licensed under the BSD 3-Clause License. See NOTICES for details.

"""
Tests for comparing Heston MC price/greeks implementation with QuantLib.

This module contains tests to ensure prices are consistent with baseline.
"""

import pytest

from quantlab.hedging.greeks import heston_delta_bump_revalue
from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess

# Optional import: skip test if QuantLib not available
ql = pytest.importorskip("QuantLib")


def heston_call_price_QL(
    K=100,
    T=1.0,
    r=0.0,
    q=0.0,
    kappa=1.0,
    theta=0.04,
    eta=0.3,
    rho=-0.5,
    v0=0.04,
    S0=100,
):
    """Quantlib Heston call price for given parameters."""
    # QuantLib setup
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today

    # Yield curves
    risk_free = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    dividend = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, ql.Actual365Fixed())
    )

    # Heston process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    heston_process = ql.HestonProcess(
        risk_free, dividend, spot_handle, v0, kappa, theta, eta, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    # engine = ql.COSHestonEngine(heston_model, L=15)

    option = ql.EuropeanOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K),
        ql.EuropeanExercise(today + ql.Period(int(T * 365), ql.Days)),
    )
    option.setPricingEngine(engine)
    ql_price = option.NPV()
    return ql_price


def prices_close(a, b, abs_tol=1e-4, rel_tol=1e-3):
    """Values comparison function."""
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


@pytest.mark.parametrize(
    "kappa,theta,eta,rho,v0,K,T",
    [
        (15.7306, 0.1339, 0.2514, 0.1637, 0.0770, 1.0, 0.25),  # ATM
    ],
)
def test_heston_delta_bump_revalue_vs_quantlib_normalized_parametrized(
    kappa, theta, eta, rho, v0, K, T
):
    """Test closeness of prices in normalized setup."""
    market_state = MarketState(stock_price=1.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=v0, kappa=kappa, theta=theta, eta=eta, rho=rho)
    process = HestonProcess(params, market_state)
    option = StockOption(strike_price=K, expiration_time=T, is_call=True)

    # price = heston_euler_mc_price(option, process, n_paths=500000, n_steps=1000)

    # Compute delta
    delta, C0, C_up, C_down = heston_delta_bump_revalue(
        option,
        process,
        n_paths=200000,  # increase for stable Greeks
        n_steps=1000,  # high steps for accuracy
        bump_size=0.01,  # 1% bump (0.01 on S0=1.0)
        seed=42,
    )

    print(f"MC option price: {C0:.6f}")
    print(f"Delta:        {delta:.6f}")

    # print("MC price:", price)

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
        S0=1.0,
    )
    print("QL price:", price_ql)

    assert prices_close(C0, price_ql, abs_tol=1e-3)
