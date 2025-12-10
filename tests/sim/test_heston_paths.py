# tests/sim/test_heston_paths.py
import pytest
import torch
import numpy as np
from quantlab.sim.heston.paths import simulate_heston_paths_numpy, simulate_heston_paths_torch

def test_heston_paths_shape_numpy():
    S, v = simulate_heston_paths_numpy(
            S0=100.0, v0=0.04, r=0.00, q=0.0, 
            kappa=1.0, theta=0.04, eta=0.3, rho=-0.7, T=1.0, 
            N=1000, M=50
    )
    assert S.shape == (1000, 51)
    assert v.shape == (1000, 51)
    assert np.all(v >= 0)  # variance can't be negative

def test_heston_paths_shape_torch():
    S, v = simulate_heston_paths_torch(
            S0=100.0, v0=0.04, r=0.00, q=0.0,
            kappa=1.0, theta=0.04, eta=0.3, rho=-0.7, T=1.0,
            N=1000, M=50
    )
    assert S.shape == (1000, 51)
    assert v.shape == (1000, 51)
    assert torch.all(v >= 0)  # variance can't be negative

