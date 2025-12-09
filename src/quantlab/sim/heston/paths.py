import torch
import numpy as np


def simulate_heston_paths(S0: float, v0: float, r: float, q: float, kappa: float, theta: float, eta: float, rho: float, T: float, N: int, M: int, device: str = 'cpu'):
    dt = T / M
    Z1 = torch.randn(N, M, device=device)
    Z2 = torch.randn(N, M, device=device)

    dW = Z1 * np.sqrt(dt)  # Brownian motion for S
    dB = (rho * Z1 + np.sqrt(1 - rho**2)*Z2) * np.sqrt(dt)  # Brownian motion for v

    S = torch.empty(N, M + 1, device=device)
    S[:,0] = S0
    v = torch.empty(N, M + 1, device=device)
    v[:,0] = v0

    for t in range(M):
        drift_v = kappa * (theta - v[:, t]) * dt
        diffusion_v = eta * np.sqrt(v[:, t]) * dB[:, t]
        v[:, t+1] = v[:, t] + drift_v + diffusion_v
        v[:, t+1] = np.maximum(v[:, t+1], 0)  # Ensure non-negative

        drift_S = (r - q) * S[:, t] * dt
        diffusion_S = np.sqrt(v[:, t]) * S[:, t] * dW[:, t]
        S[:, t+1] = S[:, t] + drift_S + diffusion_S

    return S, v




