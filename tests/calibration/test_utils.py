"""Tests for calibration utilities module."""
import numpy as np

from quantlab.calibration.utils import calculate_vega_weights


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
