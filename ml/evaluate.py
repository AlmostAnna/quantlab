"""
Evaluating HedgeNet.

This module contains implementation of HedgeNet evaluation pipeline.
"""

import torch

from ml.config import GBMConfig, HedgingConfig, StressTestConfig
from ml.metrics.pnl import compute_pnl_with_tx
from ml.models.hedge_net import HedgeNet
from ml.sim.gbm import simulate_gbm


def load_model(hidden_dim, path="artifacts/hedge_net_tx.pth", device="cpu"):
    """Load HedgeNet model from the file."""
    net = HedgeNet(hidden_dim).to(device)
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def prepare_inputs_for_model(S, K, T, M, device="cpu"):
    """Prepare inputs for HedgeNet evaluation."""
    t_grid = torch.linspace(0, T - T / M, M, device=device)
    tau = T - t_grid
    N = S.size(0)
    tau_batch = tau.unsqueeze(0).expand(N, -1).reshape(-1)
    moneyness_batch = (S[:, :-1] / K).reshape(-1)
    return tau_batch, moneyness_batch, N, M


def run_stress_test():
    """Run some testing scenarios on hedging with HedgeNet."""
    base_gbm = GBMConfig()
    base_hedge = HedgingConfig()
    stress_cfg = StressTestConfig()

    net = load_model(base_hedge.hidden_dim, device=base_hedge.device)

    results = {"sigma": [], "lambda_tx": [], "M": [], "mean_abs_pnl": [], "std_pnl": []}

    # Vary sigma
    for sigma in stress_cfg.sigma_vals:
        gbm = GBMConfig(S0=base_gbm.S0, sigma=sigma, T=base_gbm.T, N=5000, M=base_gbm.M)
        S = simulate_gbm(**gbm.__dict__, device=base_hedge.device).float()
        tau_flat, moneyness_flat, N, M = prepare_inputs_for_model(
            S, base_hedge.K, gbm.T, gbm.M
        )
        with torch.no_grad():
            phi_flat = net(tau_flat, moneyness_flat)
            phi = phi_flat.reshape(N, M)
            pnl = compute_pnl_with_tx(S, base_hedge.K, phi, base_hedge.lambda_tx)
        results["sigma"].append(sigma)
        results["lambda_tx"].append(base_hedge.lambda_tx)
        results["M"].append(gbm.M)
        results["mean_abs_pnl"].append(pnl.abs().mean().item())
        results["std_pnl"].append(pnl.std().item())

    # Vary lambda_tx
    for lam in stress_cfg.lambda_vals:
        S = simulate_gbm(**base_gbm.__dict__, device=base_hedge.device).float()
        tau_flat, moneyness_flat, N, M = prepare_inputs_for_model(
            S, base_hedge.K, base_gbm.T, base_gbm.M
        )
        with torch.no_grad():
            phi_flat = net(tau_flat, moneyness_flat)
            phi = phi_flat.reshape(N, M)
            pnl = compute_pnl_with_tx(S, base_hedge.K, phi, lam)
        results["sigma"].append(base_gbm.sigma)
        results["lambda_tx"].append(lam)
        results["M"].append(base_gbm.M)
        results["mean_abs_pnl"].append(pnl.abs().mean().item())
        results["std_pnl"].append(pnl.std().item())

    # Vary M (rebalancing frequency)
    for M in stress_cfg.M_vals:
        gbm = GBMConfig(S0=base_gbm.S0, sigma=base_gbm.sigma, T=base_gbm.T, N=5000, M=M)
        S = simulate_gbm(**gbm.__dict__, device=base_hedge.device).float()
        tau_flat, moneyness_flat, N, M_actual = prepare_inputs_for_model(
            S, base_hedge.K, gbm.T, gbm.M
        )
        with torch.no_grad():
            phi_flat = net(tau_flat, moneyness_flat)
            phi = phi_flat.reshape(N, M_actual)
            pnl = compute_pnl_with_tx(S, base_hedge.K, phi, base_hedge.lambda_tx)
        results["sigma"].append(base_gbm.sigma)
        results["lambda_tx"].append(base_hedge.lambda_tx)
        results["M"].append(M)
        results["mean_abs_pnl"].append(pnl.abs().mean().item())
        results["std_pnl"].append(pnl.std().item())

    return results
