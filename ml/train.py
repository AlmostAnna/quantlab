# ml/train.py
import os
import argparse

import torch
from ml.config import GBMConfig, HestonConfig, HedgingConfig
from quantlab.models.gbm import simulate_gbm
from quantlab.models.heston import simulate_heston_paths

from ml.models.hedge_net import HedgeNet
from ml.metrics.pnl import compute_pnl_with_tx

def prepare_inputs(S, K, T, M, device='cpu'):
    t_grid = torch.linspace(0, T - T/M, M, device=device)
    tau_grid = T - t_grid
    N = S.size(0)
    tau_batch = tau_grid.unsqueeze(0).expand(N, -1)
    moneyness_batch = S[:, :-1] / K
    return tau_batch.reshape(-1), moneyness_batch.reshape(-1)

def main(args):

    umodel = args.data_type_model
    if umodel=='heston':
        umodel_cfg = HestonConfig()
    else:
        umodel_cfg = GBMConfig()

    hedge_cfg = HedgingConfig()

    print(f"Simulating {umodel} paths ...")
    if umodel=='heston':
        S,_ = simulate_heston_paths(**umodel_cfg.__dict__, device=hedge_cfg.device)
        S = S.float()
    else:
        S = simulate_gbm(**umodel_cfg.__dict__, device=hedge_cfg.device).float()
    tau_flat, moneyness_flat = prepare_inputs(S, hedge_cfg.K, umodel_cfg.T, umodel_cfg.M, hedge_cfg.device)

    net = HedgeNet(hedge_cfg.hidden_dim).to(hedge_cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=hedge_cfg.lr)

    print("Training hedging network...")
    for epoch in range(hedge_cfg.epochs):
        optimizer.zero_grad()
        phi_flat = net(tau_flat, moneyness_flat)
        phi = phi_flat.reshape(S.size(0), -1)
        pnl = compute_pnl_with_tx(S, hedge_cfg.K, phi, hedge_cfg.lambda_tx)
        loss = torch.mean(pnl ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(net.state_dict(), "hedge_net_tx.pth")
    print("Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Hedging")
    parser.add_argument("--data_type_model", type=str, default='gbm')
    
    args = parser.parse_args()
    main(args)


