"""Tests for calibration utilities module."""
from unittest.mock import patch

import numpy as np
import pytest
from vollib.black import undiscounted_black

from quantlab.calibration.utils import (
    calculate_vega_weights,
    safe_implied_volatility_of_undiscounted_price,
)


def test_calculate_vega_weights():
    """Test vega weights calculation produces correct shape and behavior."""
    # Setup: ATM + OTM + ITM options
    strikes = np.array([90.0, 100.0, 110.0])  # ATM at 100
    maturities = np.array([1.0, 1.0, 1.0])  # Same maturity
    implied_vols = np.array([0.2, 0.2, 0.2])  # Same vol
    forwards = np.array([100.0, 100.0, 100.0])  # Forward = 100
    interest_rates = np.array([0.02, 0.02, 0.02])

    weights = calculate_vega_weights(
        strikes=strikes,
        maturities=maturities,
        implied_vols=implied_vols,
        forwards=forwards,
        interest_rates=interest_rates,
        min_weight=1e-6,
    )

    # Assertions
    assert len(weights) == 3, "Should return 3 weights for 3 options"
    assert all(w > 0 for w in weights), "All weights should be positive"
    assert all(w >= 1e-6 for w in weights), "Weights should respect minimum"

    # ATM should have highest vega (highest sensitivity to vol)
    assert weights[1] > weights[0], "ATM should have higher vega than OTM (put)"
    assert weights[1] > weights[2], "ATM should have higher vega than OTM (call)"

    # Verify weights are reasonable magnitude (not NaN/inf)
    assert not np.any(np.isnan(weights)), "Weights should not be NaN"
    assert not np.any(np.isinf(weights)), "Weights should not be infinite"
    assert np.all(np.isfinite(weights)), "All weights should be finite"


def test_calculate_vega_weights_minimum_bound():
    """Test that minimum weight is respected."""
    # Degenerate case: very short maturity (vega approaches 0)
    strikes = np.array([100.0])
    maturities = np.array([1e-6])  # Very short maturity
    implied_vols = np.array([0.01])  # Low vol
    forwards = np.array([100.0])
    interest_rates = np.array([0.02])

    weights = calculate_vega_weights(
        strikes=strikes,
        maturities=maturities,
        implied_vols=implied_vols,
        forwards=forwards,
        interest_rates=interest_rates,
        min_weight=0.05,  # High minimum
    )

    # Should return the minimum, not the tiny calculated vega
    expected_min = 0.05
    assert (
        weights[0] == expected_min
    ), f"Should return minimum weight {expected_min}, got {weights[0]}"


def test_calculate_vega_weights_different_maturities():
    """Test vega increases with maturity (longer options more sensitive to vol)."""
    strikes = np.array([100.0, 100.0, 100.0])  # All ATM
    maturities = np.array([0.25, 1.0, 2.0])  # Increasing maturity
    implied_vols = np.array([0.2, 0.2, 0.2])  # Same vol
    forwards = np.array([100.0, 100.0, 100.0])
    interest_rates = np.array([0.02, 0.02, 0.02])

    weights = calculate_vega_weights(
        strikes=strikes,
        maturities=maturities,
        implied_vols=implied_vols,
        forwards=forwards,
        interest_rates=interest_rates,
        min_weight=1e-6,
    )

    # Longer maturity should have higher vega
    assert weights[0] < weights[1] < weights[2], "Vega should increase with maturity"


PRIMARY_IV = (
    "quantlab.calibration.utils.implied_volatility_of_undiscounted_option_price"
)

F, T = 100.0, 1.0
# The fallback stops at |price error| < 1e-8. At the deep-ITM/OTM low-vol corner
# vega ~ 1.7e-3, so 1e-8 in price space is ~6e-6 in vol space -> assert abs=1e-5.
IV_TOL = 1e-5


@pytest.mark.parametrize("flag", ["c", "p"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])  # ITM / ATM / OTM
@pytest.mark.parametrize("sigma", [0.05, 0.20, 0.80])
def test_fallback_bisection_recovers_known_vol(sigma, K, flag):
    """Test that primary raises -> bisection still inverts price back to the true vol."""  # noqa: E501
    price = undiscounted_black(F, K, sigma, T, flag)
    with patch(PRIMARY_IV, side_effect=RuntimeError("force fallback")) as primary:
        iv = safe_implied_volatility_of_undiscounted_price(price, F, K, T, flag)
    primary.assert_called_once()  # prove we actually exercised the fallback
    assert iv == pytest.approx(sigma, abs=IV_TOL)


@pytest.mark.parametrize(
    "bad_output", [float("nan"), 1e308, -1e308], ids=["nan", "above_max", "below_min"]
)
def test_fallback_triggered_by_invalid_primary_output(bad_output):
    """Test that NaN / Jaeckel-style sentinels are treated as failure, not returned as IV."""  # noqa: E501
    price = undiscounted_black(F, 100.0, 0.20, T, "c")
    with patch(PRIMARY_IV, return_value=bad_output):
        iv = safe_implied_volatility_of_undiscounted_price(price, F, 100.0, T, "c")
    assert iv == pytest.approx(0.20, abs=IV_TOL)


@pytest.mark.parametrize(
    "flag,Fx,Kx",
    [("c", 100.0, 100.0), ("c", 110.0, 100.0), ("p", 100.0, 100.0), ("p", 90.0, 100.0)],
)
def test_fallback_near_intrinsic_returns_min_iv(flag, Fx, Kx):
    """Test that at intrinsic there is no meaningful IV -> min_iv (guard, not bisection)."""  # noqa: E501
    intrinsic = max(Fx - Kx, 0.0) if flag == "c" else max(Kx - Fx, 0.0)
    with patch(PRIMARY_IV, side_effect=RuntimeError("force fallback")):
        iv = safe_implied_volatility_of_undiscounted_price(
            intrinsic + 1e-12, Fx, Kx, T, flag
        )
    assert iv == 1e-6


def test_primary_success_skips_fallback():
    """Test a valid primary result is returned untouched; the fallback never runs."""
    with patch(PRIMARY_IV, return_value=0.1234) as primary:
        iv = safe_implied_volatility_of_undiscounted_price(5.6, F, 100.0, T, "c")
    primary.assert_called_once()
    assert iv == 0.1234
