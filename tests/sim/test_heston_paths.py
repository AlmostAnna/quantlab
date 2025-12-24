"""
Tests for Heston MC path generation.

This module contains tests to ensure basic correctness
of generated paths.
"""
import numpy as np
import torch

from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess
from quantlab.sim.heston.paths import (
    simulate_heston_paths_numpy,
    simulate_heston_paths_torch,
)


def test_heston_paths_shape_numpy():
    """Test paths shape using Numpy."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=0.04, kappa=1.0, theta=0.04, eta=0.3, rho=-0.7)
    process = HestonProcess(params, market_state)

    S, v = simulate_heston_paths_numpy(process, T=1.0, N=1000, M=50)
    assert S.shape == (1000, 51)
    assert v.shape == (1000, 51)
    assert np.all(v >= 0)  # variance can't be negative


def test_heston_paths_shape_torch():
    """Test paths shape using PyTorch."""
    market_state = MarketState(stock_price=100.0, interest_rate=0.0, time=0.0)
    params = HestonParameters(v0=0.04, kappa=1.0, theta=0.04, eta=0.3, rho=-0.7)
    process = HestonProcess(params, market_state)

    S, v = simulate_heston_paths_torch(process, T=1.0, N=1000, M=50)
    assert S.shape == (1000, 51)
    assert v.shape == (1000, 51)
    assert torch.all(v >= 0)  # variance can't be negative
