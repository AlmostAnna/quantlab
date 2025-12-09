# ml/models/hedge_net.py
import torch
import torch.nn as nn

class HedgeNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, tau, moneyness):
        x = torch.stack([tau, moneyness], dim=1)
        return self.net(x).squeeze(-1)

