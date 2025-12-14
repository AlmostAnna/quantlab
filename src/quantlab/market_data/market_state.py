"""The state of the market at a given time."""
from dataclasses import dataclass

from quantlab.utils.types import Floats


@dataclass
class MarketState:
    """
    Represents the state of the market at a given time.

    The fields can hold either a single scalar value (representing one scenario)
    or a numpy array of values (allowing for vectorized calculations across
    multiple scenarios, assets, or time points).

    Attributes:
        stock_price (Floats): The current stock price(s). Can be a float for a
                             single price or a FloatArray for multiple prices.
        interest_rate (Floats): The risk-free interest rate(s) applicable. Can be a
                              float for a single rate or a FloatArray for multiple
                              rates.
        time (Floats): The current time(s), defaulting to 0. Can be a float for a
                      single time or a FloatArray for multiple times. Defaults to 0.
    """

    stock_price: Floats
    interest_rate: Floats
    time: Floats = 0.0


# Example usage:
# Single market state:
# market_single = MarketState(stock_price=100.0, interest_rate=0.05, time=0.0)

# Multiple market states (e.g., for vectorized pricing):
# stock_prices = np.array([100.0, 105.0, 95.0])
# rates = np.array([0.05, 0.04, 0.06])
# times = np.array([0.0, 0.1, 0.2])
# market_batch = MarketState(stock_price=stock_prices, interest_rate=rates, time=times)
