"""Generic StockOption."""
from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from quantlab.utils.types import Floats


@dataclass
class StockOption:
    """
    A data class representing a European stock option.

    The strike price and expiration time can be specified as single values
    (for one option) or as numpy arrays (allowing for vectorized calculations
    across multiple options simultaneously).

    Attributes:
        strike_price (Floats): The exercise price(s) of the option(s).
                             Can be a float for a single strike or a FloatArray
                             for multiple strikes.
        expiration_time (Floats): The time(s) to expiration in years.
                                Can be a float for a single time or a FloatArray
                                for multiple times.
        is_call (Union[bool, npt.NDArray[np.bool_]]): Boolean flag(s) indicating
                                                    if the option(s) are calls (True)
                                                    or puts (False). Must be specified
                                                    during initialization.
    """

    strike_price: Floats
    expiration_time: Floats  # in years
    is_call: Union[bool, npt.NDArray[np.bool_]] = None

    def __post_init__(self):
        """Validate that the 'is_call' attribute is set after initialization."""
        assert self.is_call is not None

    def payoff(self, stock_price: Floats) -> Floats:
        """
        Calculate the payoff of the option at expiration for given stock price.

        Supports vectorized operations if the input 'stock_price' or the instance's
        attributes ('strike_price', 'is_call') are numpy arrays.

        Args:
            stock_price (Floats): The stock price(s) at expiration. Can be a float
                                  for a single price or a FloatArray
                                  for multiple prices.

        Returns:
            Floats: The payoff(s) of the option(s).
                    A float if all inputs are scalars,
                    otherwise a FloatArray.
                    Payoff = max(S - K, 0) for calls, max(K - S, 0) for puts.
        """
        call_payoff = np.maximum(0, stock_price - self.strike_price)
        put_payoff = np.maximum(0, self.strike_price - stock_price)
        return np.where(self.is_call, call_payoff, put_payoff)
