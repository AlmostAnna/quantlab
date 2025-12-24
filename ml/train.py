"""
Training Module and Command-Line Interface for Deep Hedging Model.

This module provides functions to train deep hedging neural net on
different simulated asset paths (e.g. GDM, Heston).

When run as a script (e.g., `python -m ml.train --data_type_model=gbm`), it provides a
command-line interface to execute the training functions with configurable
parameters.

Example Usage as a Script:
    # Train on GMB paths
    python -m ml.train --data_type_model=gbm

    # Train on Heston paths
    python -m ml.train --data_type_model=heston
"""
import argparse

import torch

from ml.config import GBMConfig, HedgingConfig, HestonConfig
from quantlab.ml.metrics.pnl import compute_pnl_with_tx
from quantlab.ml.models.hedge_net import HedgeNet
from quantlab.sim.gbm import simulate_gbm_torch as simulate_gbm
from quantlab.sim.heston.paths import (
    simulate_heston_paths_torch as simulate_heston_paths,
)


def prepare_inputs(S, K, T, M, device="cpu"):
    """Prepare input batches for training."""
    t_grid = torch.linspace(0, T - T / M, M, device=device)
    tau_grid = T - t_grid
    N = S.size(0)
    tau_batch = tau_grid.unsqueeze(0).expand(N, -1)
    moneyness_batch = S[:, :-1] / K
    return tau_batch.reshape(-1), moneyness_batch.reshape(-1)


def main(args):
    """Train Deep Hedging Model."""
    umodel = args.data_type_model
    if umodel == "heston":
        umodel_cfg = HestonConfig()
    else:
        umodel_cfg = GBMConfig()

    hedge_cfg = HedgingConfig()

    print(f"Simulating {umodel} paths ...")
    if umodel == "heston":
        process, sim_T, sim_N, sim_M = umodel_cfg.to_process_and_params(
            device=hedge_cfg.device
        )
        S, _ = simulate_heston_paths(
            process, sim_T, sim_N, sim_M, device=hedge_cfg.device
        )
        # S,_ = simulate_heston_paths(**umodel_cfg.__dict__, device=hedge_cfg.device)
        S = S.float()
    else:
        S = simulate_gbm(**umodel_cfg.__dict__, device=hedge_cfg.device).float()
    tau_flat, moneyness_flat = prepare_inputs(
        S, hedge_cfg.K, umodel_cfg.T, umodel_cfg.M, hedge_cfg.device
    )

    net = HedgeNet(hedge_cfg.hidden_dim).to(hedge_cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=hedge_cfg.lr)

    print("Training hedging network...")
    for epoch in range(hedge_cfg.epochs):
        optimizer.zero_grad()
        phi_flat = net(tau_flat, moneyness_flat)
        phi = phi_flat.reshape(S.size(0), -1)
        pnl = compute_pnl_with_tx(S, hedge_cfg.K, phi, hedge_cfg.lambda_tx)
        loss = torch.mean(pnl**2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(net.state_dict(), "hedge_net_tx.pth")
    print("Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Hedging")
    parser.add_argument("--data_type_model", type=str, default="gbm")

    args = parser.parse_args()
    main(args)
