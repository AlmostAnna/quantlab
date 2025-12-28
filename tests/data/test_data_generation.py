"""
Tests for data generation.

This module contains tests to ensure basic correctness
of generated data.
"""
from quantlab.data.synthetic import generate_heston_vol_surface


def test_heston_data():
    """Test data shape for Heston model data."""
    strikes, maturities, vols = generate_heston_vol_surface(seed=42)
    print(f"Generated {len(strikes)} data points")
    print(f"Vol range: {vols.min():.3f} to {vols.max():.3f}")

    assert len(strikes) == 9 * 4
    assert len(maturities) == 9 * 4
    assert len(vols) == 9 * 4
