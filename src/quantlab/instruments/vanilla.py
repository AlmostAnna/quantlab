"""
Module for vanilla instruments.

This module contains definitions of vanilla call and put options.
"""
from quantlab.instruments.base import StockOption


class CallStockOption(StockOption):
    """
    Represents a European call option.

    Equivalent to StockOption(is_call=True). The 'is_call' attribute is automatically
    set to True during initialization.
    """

    def __post_init__(self):
        """Set is_call to True."""
        self.is_call = True


class PutStockOption(StockOption):
    """
    Represents a European put option.

    Equivalent to StockOption(is_call=False). The 'is_call' attribute is automatically
    set to False during initialization.
    """

    def __post_init__(self):
        """Set is_call to False."""
        self.is_call = False
