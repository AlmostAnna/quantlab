"""
Heston model based MC path simulators.

This module contains implementation of MC path simulators for Heston model.
"""
import numpy as np
import torch

from quantlab.models.heston.model import HestonProcess


# For traditional quant finance
def simulate_heston_paths_numpy(process: HestonProcess, T: float, N: int, M: int):
    """
    Simulate Heston paths.

    Args:
        process: The HestonProcess object containing model parameters
                and initial market state (S0, r, v0, etc.).
        T: The total time to simulate (e.g., option maturity).
        N: The number of paths to simulate.
        M: The number of time steps per path.

    Returns:
        A tuple (S_paths, v_paths) where S_paths and v_paths are numpy arrays
        of shape (N, M+1), containing the simulated stock prices and variances
        for N paths over M+1 time points (including t=0).
    """
    # Extract parameters
    market_state = process.market_state

    S0 = market_state.stock_price
    r = market_state.interest_rate
    q = 0.0  # TODO Add dividend yield to MarketState

    # Extract model parameters
    params = process.model_params  # HestonParameters
    kappa, theta, eta, rho, v0 = (
        params.kappa,
        params.theta,
        params.eta,
        params.rho,
        params.v0,
    )

    # np.random.seed(seed)
    dt = T / M
    Z1 = np.random.randn(N, M)
    Z2 = np.random.randn(N, M)

    dW = Z1 * np.sqrt(dt)  # Brownian motion for S
    dB = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)  # Brownian motion for v

    S = np.empty((N, M + 1))
    S[:, 0] = S0
    v = np.empty((N, M + 1))
    v[:, 0] = v0

    for t in range(M):
        drift_v = kappa * (theta - v[:, t]) * dt
        diffusion_v = eta * np.sqrt(v[:, t]) * dB[:, t]
        v[:, t + 1] = v[:, t] + drift_v + diffusion_v
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)  # Ensure non-negative

        drift_S = (r - q) * S[:, t] * dt
        diffusion_S = np.sqrt(v[:, t]) * S[:, t] * dW[:, t]
        S[:, t + 1] = S[:, t] + drift_S + diffusion_S

    return S, v


# For ML/PyTorch
def simulate_heston_paths_torch(
    process: HestonProcess, T: float, N: int, M: int, device="cpu"
):
    """
    Simulate Heston paths.

    Args:
        process: The HestonProcess object containing model parameters
                and initial market state (S0, r, v0, etc.).
        T: The total time to simulate (e.g., option maturity).
        N: The number of paths to simulate.
        M: The number of time steps per path.

    Returns:
        A tuple (S_paths, v_paths) where S_paths and v_paths are numpy arrays
        of shape (N, M+1), containing the simulated stock prices and variances
        for N paths over M+1 time points (including t=0).
    """
    # Extract parameters
    market_state = process.market_state

    S0 = market_state.stock_price
    r = market_state.interest_rate
    q = 0.0  # TODO Add dividend yield to MarketState

    # Extract model parameters
    params = process.model_params  # HestonParameters
    kappa, theta, eta, rho, v0 = (
        params.kappa,
        params.theta,
        params.eta,
        params.rho,
        params.v0,
    )

    # np.random.seed(seed)

    dt = T / M
    Z1 = torch.randn(N, M, device=device)
    Z2 = torch.randn(N, M, device=device)

    dW = Z1 * torch.sqrt(torch.tensor(dt, device=device))
    dB = (
        rho * Z1 + torch.sqrt(torch.tensor(1 - rho**2, device=device)) * Z2
    ) * torch.sqrt(torch.tensor(dt, device=device))

    S = torch.empty(N, M + 1, device=device)
    S[:, 0] = S0
    v = torch.empty(N, M + 1, device=device)
    v[:, 0] = v0

    for t in range(M):
        drift_v = kappa * (theta - v[:, t]) * dt
        diffusion_v = eta * torch.sqrt(v[:, t]) * dB[:, t]
        v[:, t + 1] = v[:, t] + drift_v + diffusion_v
        v[:, t + 1] = torch.maximum(v[:, t + 1], torch.tensor(0.0, device=device))

        drift_S = (r - q) * S[:, t] * dt
        diffusion_S = torch.sqrt(v[:, t]) * S[:, t] * dW[:, t]
        S[:, t + 1] = S[:, t] + drift_S + diffusion_S

    return S, v
