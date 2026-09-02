"""Indicadores de momentum."""
from __future__ import annotations

import pandas as pd


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (%)."""
    return series.pct_change(periods=period) * 100


def momentum(series: pd.Series, period: int = 12) -> pd.Series:
    """Momentum absoluto (diferencia de precio)."""
    return series.diff(periods=period)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (0..-100)."""
    high_max = df["High"].rolling(window=period).max()
    low_min = df["Low"].rolling(window=period).min()
    rng = (high_max - low_min).replace(0, float("nan"))
    return -100 * (high_max - df["Close"]) / rng