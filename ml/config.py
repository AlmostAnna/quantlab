"""
Configuration for Deep Hedging.

This module contains configurations needed to train and test hedging NN.
"""
from dataclasses import dataclass

from quantlab.market_data.market_state import MarketState
from quantlab.models.heston.model import HestonParameters, HestonProcess


@dataclass
class GBMConfig:
    """GBM Simulation Configuration."""

    S0: float = 100.0
    sigma: float = 0.2
    T: float = 1.0
    N: int = 20000
    M: int = 50


@dataclass
class HestonConfig:
    """Heston Model Simulation Configuration."""

    # Market State Parameters
    S0: float = 100.0
    r: float = 0.05
    q: float = 0.0  # TODO Add dividend yield to MarketState
    # Model Parameters
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    eta: float = 0.3
    rho: float = -0.7
    # Simulation Parameters
    T: float = 1.0
    N: int = 20000
    M: int = 50

    def to_process_and_params(self, device="cpu"):
        """
        Create the HestonProcess object and extract simulation parameters.

        This bridges the config to the simulation function signature.
        """
        # Create MarketState
        market_state = MarketState(
            stock_price=self.S0,
            interest_rate=self.r,
            # dividend_yield=self.q, # TODO If MarketState has dividend_yield
            time=0.0,
        )

        # Create HestonParameters
        heston_params = HestonParameters(
            v0=self.v0, kappa=self.kappa, theta=self.theta, eta=self.eta, rho=self.rho
        )

        # Create HestonProcess
        process = HestonProcess(
            model_params=heston_params, market_state=market_state, backend="torch"
        )

        # Return the process object and simulation parameters
        return process, self.T, self.N, self.M


@dataclass
class HedgingConfig:
    """Hedging Configuration."""

    # This config defines the option being hedged
    K: float = 100.0  # Strike price
    T: float = (
        1.0  # Maturity (should match simulation T for consistency, or be derived)
    )
    lambda_tx: float = 0.05  # Transaction cost parameter
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 500
    device: str = "cpu"


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""

    sigma_vals: list = None
    lambda_vals: list = None
    M_vals: list = None

    def __post_init__(self):
        """Set defaults."""
        if self.sigma_vals is None:
            self.sigma_vals = [0.15, 0.2, 0.25, 0.3]
        if self.lambda_vals is None:
            self.lambda_vals = [0.0, 0.01, 0.05, 0.1]
        if self.M_vals is None:
            self.M_vals = [10, 25, 50, 100]
