"""Indicadores técnicos principales: tendencia, momentum y volatilidad.

Todos los cálculos devuelven Series de pandas indexadas igual que el input.
Los indicadores NO producen valores en las primeras filas (NaN) cuando
no hay suficiente historia para el período.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Media móvil simple."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Media móvil exponencial."""
    return series.ewm(span=period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range de Wilder."""
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    tr = _true_range(df)
    # RMA (Wilder smoothing) = EMA con alpha = 1/period
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Índice de Fuerza Relativa (RSI) de Wilder."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Si no hay pérdidas, RSI = 100
    out = out.where(avg_loss != 0, 100.0)
    out = out.fillna(50.0)
    return out


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD, línea de señal e histograma."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - signal_line
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": hist})


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX, +DI y -DI (Wilder)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)

    def _rma(x: pd.Series, p: int) -> pd.Series:
        return x.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()

    atr_ = _rma(tr, period)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=df.index), period) / atr_
    minus_di = 100 * _rma(pd.Series(minus_dm, index=df.index), period) / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3
) -> pd.DataFrame:
    """Estocástico %K y %D."""
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    raw_k = 100 * (df["Close"] - low_min) / rng

    k = raw_k.rolling(window=smooth).mean()
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def parabolic_sar(
    df: pd.DataFrame, start_af: float = 0.02, step_af: float = 0.02, max_af: float = 0.2
) -> pd.Series:
    """Parabolic Stop and Reverse (SAR) de Wilder."""
    return _parabolic_sar_fallback(df, start_af, step_af, max_af)


def _parabolic_sar_fallback(
    df: pd.DataFrame, start_af: float, step_af: float, max_af: float
) -> pd.Series:
    """Implementación propia del SAR de Wilder."""
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)

    up_trend = True
    af = start_af
    ep = low[0]
    sar[0] = high[0]

    high_out = high.copy()
    low_out = low.copy()

    for i in range(1, n):
        if up_trend:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], low_out[i - 1], low_out[i - 2] if i >= 2 else low_out[i - 1])
            if low[i] < sar[i]:
                up_trend = False
                sar[i] = ep
                ep = low[i]
                af = start_af
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step_af, max_af)
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], high_out[i - 1], high_out[i - 2] if i >= 2 else high_out[i - 1])
            if high[i] > sar[i]:
                up_trend = True
                sar[i] = ep
                ep = high[i]
                af = start_af
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step_af, max_af)

    return pd.Series(sar, index=df.index)
