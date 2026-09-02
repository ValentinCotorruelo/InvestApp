"""Clasificación de tendencia por estructura de máximos/mínimos."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..indicators.technical import adx, ema
from .pivots import Pivot


@dataclass
class TrendContext:
    """Contexto de tendencia del activo."""

    regime: str  # "uptrend" | "downtrend" | "range" | "mix"
    pattern: str  # "ascendente" | "descendente" | "lateral" | "indefinido"
    price_above_ema200: bool
    adx_value: float
    plus_di: float
    minus_di: float

    @property
    def bullish(self) -> bool:
        return self.regime == "uptrend"

    @property
    def bearish(self) -> bool:
        return self.regime == "downtrend"


def _classify_pattern(high_pivots: list[Pivot], low_pivots: list[Pivot]) -> str:
    """Clasifica la secuencia de máximos/mínimos."""
    if len(low_pivots) >= 2 and len(high_pivots) >= 2:
        lows_up = low_pivots[-1].price > low_pivots[-2].price
        highs_up = high_pivots[-1].price > high_pivots[-2].price
        if lows_up and highs_up:
            return "ascendente"
        if (not lows_up) and (not highs_up):
            return "descendente"
        return "lateral"
    if len(low_pivots) >= 2:
        return "ascendente" if low_pivots[-1].price > low_pivots[-2].price else "descendente"
    if len(high_pivots) >= 2:
        return "ascendente" if high_pivots[-1].price > high_pivots[-2].price else "descendente"
    return "indefinido"


def detect_trend(
    df: pd.DataFrame,
    high_pivots: list[Pivot],
    low_pivots: list[Pivot],
    adx_period: int = 14,
    ema_long: int = 200,
    adx_threshold: float = 25.0,
) -> TrendContext:
    """Determina el régimen de mercado usando ADX, EMA200 y estructura."""
    adx_df = adx(df, adx_period)
    adx_val = float(adx_df["adx"].dropna().iloc[-1]) if adx_df["adx"].notna().any() else 0.0
    plus_di = float(adx_df["plus_di"].dropna().iloc[-1]) if adx_df["plus_di"].notna().any() else 0.0
    minus_di = float(adx_df["minus_di"].dropna().iloc[-1]) if adx_df["minus_di"].notna().any() else 0.0

    close = df["Close"]
    ema200 = ema(close, ema_long)
    price_above = bool(close.iloc[-1] > ema200.iloc[-1]) if ema200.notna().any() else False

    pattern = _classify_pattern(high_pivots, low_pivots)

    if adx_val >= adx_threshold:
        if pattern == "ascendente" or (price_above and plus_di > minus_di):
            regime = "uptrend"
        elif pattern == "descendente" or (not price_above and minus_di > plus_di):
            regime = "downtrend"
        else:
            regime = "mix"
    else:
        regime = "range"

    return TrendContext(
        regime=regime,
        pattern=pattern,
        price_above_ema200=price_above,
        adx_value=adx_val,
        plus_di=plus_di,
        minus_di=minus_di,
    )