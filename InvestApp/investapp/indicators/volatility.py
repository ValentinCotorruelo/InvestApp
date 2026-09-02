"""Indicadores de volatilidad: Bandas de Bollinger y Keltner."""
from __future__ import annotations

import pandas as pd

from .technical import atr, ema, sma


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bandas de Bollinger (media, superior, inferior)."""
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth}
    )


def keltner_channel(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0
) -> pd.DataFrame:
    """Canal de Keltner (EMA + ATR)."""
    mid = ema(df["Close"], period)
    atr_ = atr(df, period)
    upper = mid + multiplier * atr_
    lower = mid - multiplier * atr_
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})