# ml/sim/gbm.py
import torch
import numpy as np

def simulate_gbm_numpy(S0: float, sigma: float, T: float, N: int, M: int):
    dt = T / M
    Z = np.random.randn(N, M)
    S = np.empty((N, M + 1))
    S[:, 0] = S0
    sqrt_dt = np.sqrt(dt)
    for i in range(M):
        S[:, i+1] = S[:, i] * np.exp(-0.5 * sigma**2 * dt + sigma * sqrt_dt * Z[:, i])
    return S

def simulate_gbm_torch(S0: float, sigma: float, T: float, N: int, M: int, device: str = 'cpu'):
    dt = T / M
    Z = torch.randn(N, M, device=device)
    S = torch.empty(N, M + 1, device=device)
    S[:, 0] = S0
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))
    for i in range(M):
        S[:, i+1] = S[:, i] * torch.exp(-0.5 * sigma**2 * dt + sigma * sqrt_dt * Z[:, i])
    return S


