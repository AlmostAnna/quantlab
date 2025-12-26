# tests/models/test_heston_quantlib.py
# This module uses QuantLib (https://www.quantlib.org/),
# licensed under the BSD 3-Clause License. See NOTICES for details.

"""
Convinience functions for QuantLib.

This module contains helper functions for obtaining baselines with QuantLib.
"""


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
    ql_engine="analytic",
):
    """Quantlib Heston call price for given parameters."""
    import pytest

    ql = pytest.importorskip("QuantLib")  # This will skip if QuantLib not available
    # import QuantLib as ql

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
    if ql_engine == "analytic":
        engine = ql.AnalyticHestonEngine(heston_model)
    else:
        engine = ql.COSHestonEngine(heston_model, L=15)

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
