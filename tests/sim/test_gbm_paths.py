# tests/sim/test_gbm_paths.py
import pytest
import torch
import numpy as np
from quantlab.sim.gbm import simulate_gbm_numpy, simulate_gbm_torch


def test_gbm_paths_shape_numpy():
    S = simulate_gbm_numpy(
            S0=100.0, sigma=0.2, T=1.0, 
            N=1000, M=50
    )
    assert S.shape == (1000, 51)

def test_gbm_paths_shape_torch():
    S = simulate_gbm_torch(
            S0=100.0, sigma=0.2, T=1.0,
            N=1000, M=50
    )
    assert S.shape == (1000, 51)

