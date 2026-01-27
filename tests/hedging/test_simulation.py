"""
Tests for hedging paths simulation.

This module contains tests to ensure basic correctness
of simulated hedging paths.
"""
import numpy as np
import pytest

from quantlab.hedging.simulation import (
    build_delta_interpolator,
    simulate_heston_hedging_path,
)
from quantlab.instruments.base import StockOption
from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess


def test_simulate_heston_hedging_path_basic():
    """Test basic functionality of hedging path simulation."""
    # Set up parameters
    market_state = MarketState(stock_price=100.0, interest_rate=0.05, time=0.0)
    params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7)
    process = HestonProcess(params, market_state)

    option = StockOption(strike_price=100.0, expiration_time=1.0, is_call=True)

    # Build a simple delta interpolator
    S_grid = np.linspace(80, 120, 5)
    T_grid = np.linspace(0.1, 1.0, 5)
    delta_interp = build_delta_interpolator(
        option=option,
        process=process,
        S_grid=S_grid,
        T_grid=T_grid,
        n_paths=1000,  # Small number for fast test
        n_steps=10,
        seed=42,
    )

    # Run simulation
    pnl, stock_path, var_path, delta_path = simulate_heston_hedging_path(
        option=option,
        process=process,
        initial_price=100.0,
        delta_interpolator=delta_interp,
        n_steps=10,
        seed=42,
    )

    # Basic assertions
    assert isinstance(
        pnl, (float, np.floating)
    ), f"P&L should be a float, got {type(pnl)}"
    assert isinstance(stock_path, list), "Stock path should be a list"
    assert isinstance(var_path, list), "Variance path should be a list"
    assert isinstance(delta_path, list), "Delta path should be a list"
    assert (
        len(stock_path) == 11
    ), f"Stock path should have n_steps+1 points, got {len(stock_path)} (n_steps=10)"
    assert (
        len(var_path) == 11
    ), f"Variance path should have n_steps+1 points, got {len(var_path)} (n_steps=10)"
    assert (
        len(delta_path) == 11
    ), f"Delta path should have n_steps+1 points, got {len(delta_path)} (n_steps=10)"

    # Check paths have valid values
    assert all(
        np.isfinite(s) and s > 0 for s in stock_path
    ), "All stock prices should be finite and positive"
    assert all(
        np.isfinite(v) and v >= 0 for v in var_path
    ), "All variances should be finite and non-negative"
    assert all(np.isfinite(d) for d in delta_path), "All deltas should be finite"


def test_simulate_heston_hedging_path_deterministic():
    """Test that the same seed produces the same result."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.05, time=0.0)
    params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, eta=0.3, rho=-0.7)
    process = HestonProcess(params, market_state)

    option = StockOption(strike_price=100.0, expiration_time=1.0, is_call=True)

    S_grid = np.linspace(80, 120, 5)
    T_grid = np.linspace(0.1, 1.0, 5)
    delta_interp = build_delta_interpolator(
        option=option,
        process=process,
        S_grid=S_grid,
        T_grid=T_grid,
        n_paths=1000,
        n_steps=10,
        seed=42,
    )

    # Run twice with same seed
    pnl1, sp1, vp1, dp1 = simulate_heston_hedging_path(
        option=option,
        process=process,
        initial_price=100.0,
        delta_interpolator=delta_interp,
        n_steps=10,
        seed=123,
    )

    pnl2, sp2, vp2, dp2 = simulate_heston_hedging_path(
        option=option,
        process=process,
        initial_price=100.0,
        delta_interpolator=delta_interp,
        n_steps=10,
        seed=123,  # Same seed
    )

    # Should be identical
    assert pnl1 == pnl2, f"Same seed should produce same P&L: {pnl1} vs {pnl2}"
    assert sp1 == sp2, "Same seed should produce same stock path"
    assert vp1 == vp2, "Same seed should produce same variance path"
    assert dp1 == dp2, "Same seed should produce same delta path"


@pytest.mark.slow
def test_simulate_heston_hedging_path_extreme_conditions():
    """Test with extreme parameters that might cause numerical issues."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    # Extreme parameters that might cause issues
    # High vol-of-vol, high correlation
    params = HestonParameters(v0=0.01, kappa=10.0, theta=0.01, eta=0.5, rho=-0.9)
    process = HestonProcess(params, market_state)

    # OTM, short maturity
    option = StockOption(strike_price=80.0, expiration_time=0.1, is_call=True)

    S_grid = np.linspace(70, 130, 5)
    T_grid = np.linspace(0.01, 0.1, 5)
    delta_interp = build_delta_interpolator(
        option=option,
        process=process,
        S_grid=S_grid,
        T_grid=T_grid,
        n_paths=1000,
        n_steps=10,
        seed=42,
    )

    # Should not crash with extreme parameters
    pnl, stock_path, var_path, delta_path = simulate_heston_hedging_path(
        option=option,
        process=process,
        initial_price=100.0,
        delta_interpolator=delta_interp,
        n_steps=5,
        seed=42,
    )

    # Should still produce valid outputs
    assert isinstance(pnl, (float, np.floating))
    assert len(stock_path) == 6  # n_steps + 1
    assert all(np.isfinite(s) and s > 0 for s in stock_path)
    assert all(np.isfinite(v) and v >= 0 for v in var_path)
