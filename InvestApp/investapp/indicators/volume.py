"""Indicadores de volumen: OBV, volumen relativo, CMF, VWAP."""
from __future__ import annotations

import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = df["Close"].diff()
    direction = direction.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    return (direction * df["Volume"]).cumsum()


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volumen relativo: volumen actual / media del volumen de 'period' días."""
    vol_ma = df["Volume"].rolling(window=period).mean()
    return (df["Volume"] / vol_ma).replace([float("inf"), -float("inf")], float("nan"))


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (acumulado total)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, float("nan"))
    return cum_pv / cum_vol


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    high_low = (df["High"] - df["Low"]).replace(0, float("nan"))
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low
    mfv = mfm * df["Volume"]
    return mfv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (MFI)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    raw = typical * df["Volume"]
    flow_pos = raw.where(typical > typical.shift(1), 0.0)
    flow_neg = raw.where(typical < typical.shift(1), 0.0)

    pos_sum = flow_pos.rolling(window=period).sum()
    neg_sum = flow_neg.rolling(window=period).sum()
    ratio = (pos_sum / neg_sum.replace(0, float("nan"))).replace([float("inf"), -float("inf")], float("nan"))
    return 100 - (100 / (1 + ratio))