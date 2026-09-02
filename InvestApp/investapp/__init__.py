"""InvestApp - Analizador de inversiones con backtesting y market screener."""

from .data.fetcher import DataFetcher, MarketData
from .screener.runner import MarketScreener

__version__ = "0.1.0"

__all__ = ["DataFetcher", "MarketData", "MarketScreener"]
