"""
Delta-hedging neural network for deep hedging strategies.

This module implements HedgeNet, a feedforward neural network designed to learn
optimal hedging strategies for derivative securities. The network takes time to
maturity and moneyness as inputs and outputs the hedge ratio (delta).
"""


import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """
    Feedforward neural network for learning optimal hedging strategies.

    HedgeNet is designed for deep hedging applications where the goal is to
    minimize the P&L variance of a hedged portfolio by learning optimal
    dynamic hedging strategies. The network maps time-to-maturity and moneyness
    to hedge ratios.

    The architecture uses a sigmoid activation in the final layer to constrain
    outputs between 0 and 1, which represents the fraction of the underlying
    asset to hold in the hedging portfolio.

    Attributes:
        net (nn.Sequential): The neural network architecture consisting of
                           linear layers with ReLU activations followed by
                           a sigmoid output.

    Example:
        >>> import torch
        >>> net = HedgeNet(hidden_dim=64)
        >>> tau = torch.tensor([0.5, 0.25])  # Time to maturity
        >>> moneyness = torch.tensor([1.0, 0.9])  # S/K ratio
        >>> hedge_ratios = net(tau, moneyness)
        >>> print(hedge_ratios.shape)  # torch.Size([2])
        >>> assert torch.all(hedge_ratios >= 0) and torch.all(hedge_ratios <= 1)
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Initialize the HedgeNet neural network.

        Args:
            hidden_dim (int): Dimension of the hidden layers. Controls the
                            capacity of the neural network. Default is 64.

        Architecture:
            Input: 2 features (tau, moneyness)
            Hidden: 3 layers of size `hidden_dim` with ReLU activations
            Output: 1 neuron with sigmoid activation (hedge ratio in [0,1])
        """
        super(HedgeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: tau and moneyness
            nn.ReLU(),  # Non-linear activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # Non-linear activation
            nn.Linear(hidden_dim, 1),  # Output: single hedge ratio
            nn.Sigmoid(),  # Constrain to [0, 1] range
        )

    def forward(self, tau: torch.Tensor, moneyness: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HedgeNet.

        Computes hedge ratios based on time-to-maturity and moneyness.

        Args:
            tau (torch.Tensor): Time to maturity values. Shape (batch_size,)
                               representing remaining time until option expiry.
            moneyness (torch.Tensor): Moneyness values (S/K). Shape (batch_size,)
                                    where S is spot price and K is strike price.

        Returns:
            torch.Tensor: Hedge ratios (delta values) constrained to [0, 1].
                         Shape (batch_size,) representing the fraction of
                         underlying asset to hold for hedging.

        Raises:
            AssertionError: If tau and moneyness have different shapes.

        Note:
            - The output is constrained to [0, 1] via sigmoid activation
            - This assumes positive hedge ratios (long positions in underlying)
            - For put options or bearish strategies, consider sign adjustments
        """
        # Validate input shapes
        assert (
            tau.shape == moneyness.shape
        ), f"Input shapes must match: tau {tau.shape} vs moneyness {moneyness.shape}"

        # Stack inputs along feature dimension to create (batch_size, 2) tensor
        x = torch.stack([tau, moneyness], dim=1)  # Shape: (batch_size, 2)

        # Pass through network and squeeze final dimension
        # Output: (batch_size, 1) -> (batch_size,)
        return self.net(x).squeeze(-1)
