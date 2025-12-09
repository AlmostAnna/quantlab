# ml/metrics/pnl.py
import torch

def compute_pnl_with_tx(S, K, phi, lambda_tx):
    """S: (N, M+1), phi: (N, M)"""
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    dS = S[:, 1:] - S[:, :-1]  # (N, M)
    hedging_gain = torch.sum(phi * dS, dim=1)
    d_phi = phi[:, 1:] - phi[:, :-1]  # (N, M-1)
    # S[:, 1:-1] aligns with d_phi (trades happen between t1...t_{M-1})
    tx_cost = lambda_tx * torch.sum(torch.abs(d_phi) * S[:, 1:-1], dim=1)
    return payoff - hedging_gain - tx_cost

def decompose_pnl(S, K, phi, lambda_tx):
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    dS = S[:, 1:] - S[:, :-1]
    hedging_gain = torch.sum(phi * dS, dim=1)
    d_phi = phi[:, 1:] - phi[:, :-1]
    tx_cost = lambda_tx * torch.sum(torch.abs(d_phi) * S[:, 1:-1], dim=1)
    total_pnl = payoff - hedging_gain - tx_cost
    return total_pnl, hedging_gain, tx_cost

