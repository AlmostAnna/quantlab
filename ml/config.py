# ml/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class GBMConfig:
    S0: float = 100.0
    sigma: float = 0.2
    T: float = 1.0
    N: int = 20000
    M: int = 50

@dataclass
class HedgingConfig:
    K: float = 100.0
    lambda_tx: float = 0.05
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 500
    device: str = 'cpu'

@dataclass
class StressTestConfig:
    sigma_vals: list = None
    lambda_vals: list = None
    M_vals: list = None

    def __post_init__(self):
        if self.sigma_vals is None:
            self.sigma_vals = [0.15, 0.2, 0.25, 0.3]
        if self.lambda_vals is None:
            self.lambda_vals = [0.0, 0.01, 0.05, 0.1]
        if self.M_vals is None:
            self.M_vals = [10, 25, 50, 100]


