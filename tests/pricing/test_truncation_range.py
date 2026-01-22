"""
Tests for COS truncation methods.

This module contains tests to ensure basic correctness of truncation interval
calculation.
"""

import numpy as np
import pytest

from quantlab.models.heston.model import HestonParameters
from quantlab.pricing.heston.cos import (
    calc_trunc_range,
    heston_cumulants,
    log_price_moments,
    trunc_range_cumulant,
    trunc_range_jp,
)


def test_heston_cumulants_self_consistency():
    """
    Test that cumulants behave reasonably (monotonicity, signs, etc.).

    This doesn't require knowing the 'true' value, just that the function behaves.
    """
    p = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt = 1.0

    c1, c2, c4 = heston_cumulants(dt, p)

    # Basic sanity checks
    assert np.isfinite(c1), f"c1 is not finite: {c1}"
    assert np.isfinite(c2), f"c2 is not finite: {c2}"
    assert np.isfinite(c4), f"c4 is not finite: {c4}"

    assert c2 > 0, f"c2 (variance) should be positive: {c2}"

    # Test that increasing time increases variance (usually)
    dt_longer = 2.0
    _, c2_longer, _ = heston_cumulants(dt_longer, p)
    # This might not *always* hold due to mean reversion, but usually true.
    # A more robust check might be needed.
    assert (
        c2_longer >= c2
    ), f"c2 should increase with time (usually): {c2_longer} vs {c2}"


def test_trunc_range_sensitivity():
    """Test that truncation range changes reasonably with parameters."""
    p_base = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt = 1.0

    range_base = trunc_range_cumulant(dt, p_base, method="c4", L=10)

    # Increase volatility -> increase range
    p_high_vol = HestonParameters(
        kappa=2.0, theta=0.04, eta=0.5, rho=-0.7, v0=0.04
    )  # Higher eta
    range_high_vol = trunc_range_cumulant(dt, p_high_vol, method="c4", L=10)

    assert (
        range_high_vol >= range_base
    ), f"Range should increase with volatility: {range_high_vol} vs {range_base}"

    # Increase time -> increase range
    range_longer = trunc_range_cumulant(2.0, p_base, method="c4", L=10)
    assert (
        range_longer >= range_base
    ), f"Range should increase with time: {range_longer} vs {range_base}"


def test_heston_cumulants_vs_known_formula():
    """Test Heston cumulants against known analytical expressions."""
    # Standard Heston parameters
    p = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt = 1.0

    c1, c2, c4 = heston_cumulants(dt, p)

    # Known analytical expressions for Heston cumulants
    # c1 = (theta/kappa) * (dt - (1-exp(-k*dt))/k) - 0.5*v0*(1-exp(-k*dt))/k

    # Manually compute expected c1
    kdt = p.kappa * dt
    exp_kdt = np.exp(-kdt)
    int1 = (1.0 - exp_kdt) / p.kappa
    c1_expected = (p.theta / p.kappa) * (dt - int1) - 0.5 * p.v0 * int1

    # Assertions
    assert abs(c1 - c1_expected) < 1e-10, f"c1 mismatch: {c1} vs {c1_expected}"

    # c4 should be computed too (check it's finite and reasonable)
    assert np.isfinite(c4), f"c4 is not finite: {c4}"
    assert c4 >= -10.0 * c2**2, f"c4 violates bound: {c4} < {-10.0 * c2**2}"


def test_heston_cumulants_small_time_limit():
    """
    Test cumulants converge to correct limits as dt -> 0.

    For small dt: X ~ N((r-q-v0/2)*dt, v0*dt), so
    c1 -> (r-q-v0/2)*dt, c2 -> v0*dt, c4 -> 0.
    Ignore drift here (log(S_T/S_0) without rate adjustment).
    """
    p = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt_small = 1e-4

    c1, c2, c4 = heston_cumulants(dt_small, p)

    # Expected: c1 ~ -0.5*v0*dt, c2 ~ v0*dt, c4 ~ 0
    c1_expected = -0.5 * p.v0 * dt_small
    c2_expected = p.v0 * dt_small
    c4_expected = 0.0

    tol = 1e-6  # Allow for small time approximations
    assert abs(c1 - c1_expected) < tol, f"c1 small-t limit: {c1} vs {c1_expected}"
    assert abs(c2 - c2_expected) < tol, f"c2 small-t limit: {c2} vs {c2_expected}"
    assert abs(c4 - c4_expected) < tol, f"c4 small-t limit: {c4} vs {c4_expected}"


def test_trunc_range_methods_consistency():
    """Test that different truncation methods produce consistent results."""
    p = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt = 1.0

    # Cumulant method (default c4)
    trunc_c4 = trunc_range_cumulant(dt, p, method="c4", L=10)

    # Cumulant method (c2 only)
    trunc_c2 = trunc_range_cumulant(dt, p, method="c2", L=12)

    # JP method
    trunc_jp = trunc_range_jp(dt, p, epsilon=1e-6, n=4, K_bound=1.0)

    # All should be positive and finite
    assert trunc_c4 > 0 and np.isfinite(trunc_c4), f"trunc_c4 invalid: {trunc_c4}"
    assert trunc_c2 > 0 and np.isfinite(trunc_c2), f"trunc_c2 invalid: {trunc_c2}"
    assert trunc_jp > 0 and np.isfinite(trunc_jp), f"trunc_jp invalid: {trunc_jp}"

    # c4-based should generally be smaller than c2-based (L=10 vs 12)
    # or at least not wildly different
    assert (
        trunc_c4 / trunc_c2 < 3.0
    ), f"c4/c2 ratio seems too large: {trunc_c4 / trunc_c2}"


def test_calc_trunc_range_returns_endpoints():
    """Test that endpoints are reasonable."""
    p = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt = 1.0

    a, b = calc_trunc_range(p, dt, method="cumulant", cumulant_method="c4")

    assert isinstance(a, (float, np.floating, np.ndarray)), "a should be numeric"
    assert isinstance(b, (float, np.floating, np.ndarray)), "b should be numeric"
    assert b > a, f"b ({b}) should be greater than a ({a})"
    assert np.isfinite(a) and np.isfinite(b), f"a and b should be finite: ({a}, {b})"


def test_log_price_moments():
    """
    Test that log_price_moments returns correct values based on cumulants.

    mu2 = c2, mu4 = c4 + 3*c2^2.
    """
    p = HestonParameters(kappa=2.0, theta=0.04, eta=0.3, rho=-0.7, v0=0.04)
    dt = 1.0

    c1, c2, c4 = heston_cumulants(dt, p)

    mu2 = log_price_moments(dt, p, n=2)
    mu4 = log_price_moments(dt, p, n=4)

    assert abs(mu2 - c2) < 1e-10, f"mu2 mismatch: {mu2} vs {c2}"
    expected_mu4 = c4 + 3 * c2**2
    assert abs(mu4 - expected_mu4) < 1e-10, f"mu4 mismatch: {mu4} vs {expected_mu4}"

    # Test unsupported moment raises error
    with pytest.raises(ValueError):
        log_price_moments(dt, p, n=3)


def test_truncation_range_positive_definite():
    """Test that truncation range is always positive."""
    # Extreme parameters that might cause numerical issues
    extreme_params = [
        HestonParameters(
            kappa=100, theta=0.001, eta=0.01, rho=-0.99, v0=0.001
        ),  # Fast mean-rev, low vol
        HestonParameters(
            kappa=0.1, theta=2.0, eta=1.0, rho=0.99, v0=1.0
        ),  # Slow mean-rev, high vol
    ]

    for p in extreme_params:
        dt = 0.1  # Short time
        try:
            range_c4 = trunc_range_cumulant(dt, p, method="c4", L=10)
            range_jp = trunc_range_jp(dt, p, epsilon=1e-6, n=4, K_bound=1.0)
            assert range_c4 > 0, f"C4 range is not positive: {range_c4}"
            assert range_jp > 0, f"JP range is not positive: {range_jp}"
        except Exception as e:
            # If it fails due to numerical instability, that's acceptable
            # but should be documented. For now, let's just assert it passes.
            # If you expect certain param ranges to fail, catch and assert accordingly.
            raise AssertionError(f"Truncation failed for extreme params {p}: {e}")
