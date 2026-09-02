"""Clase base de estrategias + pipeline de decisiones.

Cada estrategia:
1. `prepare(df)`: precomputa features/indicadores y estructura de mercado.
2. `buy_signal(i)`: evalúa si hay punto de compra en el día i de la serie.
3. `exit_signal(i)`: evalúa si la posición debería cerrarse (señal de venta).
4. `exit_levels(i)`: devuelve SL/TP/trailing para la posición.

El screener evalúa el último índice; el backtest recorre todos.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..indicators.technical import (
    adx,
    atr,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)
from ..indicators.volume import obv, relative_volume
from ..indicators.volatility import bollinger_bands
from ..market_structure.fibonacci import compute_fibonacci
from ..market_structure.pivots import find_pivots
from ..market_structure.support_resistance import build_levels, detect_support_resistance
from ..market_structure.trend import detect_trend
from ..risk.risk_management import RiskManagementParams


@dataclass
class Signal:
    """Señal de compra generada por una estrategia."""

    side: str = "long"
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: Optional[float] = None
    reasons: list[str] = field(default_factory=list)
    strength: float = 0.0  # 0..1
    meta: dict = field(default_factory=dict)


@dataclass
class ExitInfo:
    """Niveles de salida para una posición."""

    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_mult: Optional[float] = None
    use_trailing: bool = False


@dataclass
class Evaluation:
    """Resultado de evaluar la estrategia en un día."""

    signal: Optional[Signal] = None  # Señal de compra activa
    should_exit: bool = False  # Señal de venta (salir si hay posición)
    exit_fresh: bool = False  # La señal de venta es NUEVA (no estado crónico)
    context: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _confirmed_by(pivot, date: pd.Timestamp) -> bool:
    """¿El pivote es conocido ('confirmado') antes o en 'date'?"""
    confirmed = pivot.confirmed_at if pivot.confirmed_at is not None else pivot.index
    return confirmed <= date


class Strategy(ABC):
    """Interfaz común de estrategia."""

    name = "base"
    description = ""

    def __init__(
        self,
        risk: Optional[RiskManagementParams] = None,
        min_bars_required: int = 60,
    ):
        self.risk = risk or RiskManagementParams()
        self.min_bars_required = min_bars_required
        self._df: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._high_pivots: list = []
        self._low_pivots: list = []
        self._pivots: list = []
        self._sr_levels: list = []
        self._sr_cache: dict = {}
        self._trend = None
        self._last_swing = None

    # ---- Pipeline de decisión ----

    @abstractmethod
    def _buy_logic(self, i: int) -> Optional[Signal]:
        """Reglas de compra para el día i. Debe implementarse en cada estrategia."""

    @abstractmethod
    def _exit_logic(self, i: int) -> tuple[bool, list[str]]:
        """Reglas de venta para el día i. Devuelve (debe_salir, razones)."""

    @abstractmethod
    def _exit_levels(self, i: int) -> ExitInfo:
        """Niveles de salida (SL/TP/trailing) para el día i."""

    # ---- Preparación ----

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precomputa features e información de mercado. Devuelve el df con features."""
        df = df.copy()
        self._df = df
        close = df["Close"]

        # Medias
        df["SMA20"] = sma(close, 20)
        df["EMA20"] = ema(close, 20)
        df["EMA50"] = ema(close, 50)
        df["EMA200"] = ema(close, 200)

        # Volatilidad / momentum
        df["ATR"] = atr(df, self.risk.atr_period)
        df["RSI"] = rsi(close, 14)

        macd_df = macd(close)
        df["MACD"] = macd_df["macd"]
        df["MACD_SIGNAL"] = macd_df["signal"]
        df["MACD_HIST"] = macd_df["hist"]

        adx_df = adx(df)
        df["ADX"] = adx_df["adx"]
        df["PLUS_DI"] = adx_df["plus_di"]
        df["MINUS_DI"] = adx_df["minus_di"]

        stoch_df = stochastic(df)
        df["STOCH_K"] = stoch_df["k"]
        df["STOCH_D"] = stoch_df["d"]

        # Volumen
        df["OBV"] = obv(df)
        df["REL_VOL"] = relative_volume(df)

        # Volatilidad
        bb = bollinger_bands(close)
        df["BB_MID"] = bb["mid"]
        df["BB_UPPER"] = bb["upper"]
        df["BB_LOWER"] = bb["lower"]

        # Estructura de mercado (usando todo el histórico disponible)
        pivots = find_pivots(df, left=5, right=5, min_move_pct=0.0)
        self._pivots = pivots
        self._high_pivots = [p for p in pivots if p.kind == "high"]
        self._low_pivots = [p for p in pivots if p.kind == "low"]

        atr_series = df["ATR"]
        sr_levels = detect_support_resistance(df, atr_series)
        self._sr_levels = sr_levels
        self._sr_cache = {}

        if pivots:
            try:
                self._trend = detect_trend(
                    df, self._high_pivots, self._low_pivots
                )
            except Exception:
                self._trend = None

        self._features = df
        self._ffilled = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self._ffilled[col] = df[col].ffill()
        return df

    def _f(self, i: int) -> pd.DataFrame:
        """Slice de features hasta el día i (evita look-ahead)."""
        if self._features is None:
            raise RuntimeError("Debes llamar a prepare() antes de evaluar.")
        # i puede ser int posicional o índice negativo
        idx = self._features.index[i]
        return self._features.loc[:idx]

    def _val(self, col: str, i: int, default: float = np.nan) -> float:
        """Último valor no-NaN de 'col' hasta el día i (O(1) vía ffill)."""
        arr = self._ffilled[col].to_numpy()
        if i < 0:
            i = len(arr) + i
        if i < 0 or i >= len(arr):
            return default
        v = arr[i]
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        return default

    def _crossed_above(self, fast_col: str, slow_col: str, i: int) -> bool:
        """¿'fast' cruzó por encima de 'slow' exactamente en el día i?"""
        if i < 1:
            return False
        f = self._features[fast_col].to_numpy()
        s = self._features[slow_col].to_numpy()
        prev_f, prev_s = f[i - 1], s[i - 1]
        cur_f, cur_s = f[i], s[i]
        if np.isnan(prev_f) or np.isnan(prev_s) or np.isnan(cur_f) or np.isnan(cur_s):
            return False
        return prev_f <= prev_s and cur_f > cur_s

    def _crossed_below(self, fast_col: str, slow_col: str, i: int) -> bool:
        """¿'fast' cruzó por debajo de 'slow' exactamente en el día i?"""
        if i < 1:
            return False
        f = self._features[fast_col].to_numpy()
        s = self._features[slow_col].to_numpy()
        prev_f, prev_s = f[i - 1], s[i - 1]
        cur_f, cur_s = f[i], s[i]
        if np.isnan(prev_f) or np.isnan(prev_s) or np.isnan(cur_f) or np.isnan(cur_s):
            return False
        return prev_f >= prev_s and cur_f < cur_s

    def _crossed_zero_above(self, col: str, i: int) -> bool:
        if i < 1:
            return False
        c = self._features[col].to_numpy()
        return c[i - 1] <= 0 < c[i] if not np.isnan(c[i - 1]) and not np.isnan(c[i]) else False

    def _any_within(self, func, i: int, window: int) -> bool:
        """True si 'func(j)' devuelve True para algún j en [i-window+1, i]."""
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            try:
                if func(j):
                    return True
            except Exception:
                continue
        return False

    def _pivots_before(self, date: pd.Timestamp) -> list:
        """Pivotes conocibles hasta 'date' (respetando confirmación sin mirar futuro)."""
        return [p for p in self._pivots if _confirmed_by(p, date)]

    def _high_pivots_before(self, date: pd.Timestamp) -> list:
        return [p for p in self._high_pivots if _confirmed_by(p, date)]

    def _low_pivots_before(self, date: pd.Timestamp) -> list:
        return [p for p in self._low_pivots if _confirmed_by(p, date)]

    def _sr_levels_at(self, date: pd.Timestamp) -> list:
        """Niveles S/R causales conocidos hasta 'date' (con caché por día)."""
        cached = self._sr_cache.get(date)
        if cached is not None:
            return cached
        known = self._pivots_before(date)
        pos = self._features.index.get_loc(date)
        atr_val = self._val("ATR", pos, default=0.0)
        levels = build_levels(known, atr_val, min_touches=2, max_levels=6, ref_date=date)
        self._sr_cache[date] = levels
        return levels

    def _nearest_sr_below(self, price: float, kind: str, date: pd.Timestamp):
        """Nivel S/R más cercano por debajo del precio (kinf 'support'/'resistance')."""
        candidates = [
            lvl
            for lvl in self._sr_levels_at(date)
            if lvl.kind == kind and lvl.price < price
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda lvl: lvl.price)

    def _nearest_sr_above(self, price: float, kind: str, date: pd.Timestamp):
        candidates = [
            lvl
            for lvl in self._sr_levels_at(date)
            if lvl.kind == kind and lvl.price > price
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda lvl: lvl.price)

    def _last_swing_for(self, direction: str, date: pd.Timestamp):
        """Último swing relevante (dirección 'up'/'down') antes de 'date'."""
        from ..market_structure.pivots import find_swings

        pivots_before = self._pivots_before(date)
        if len(pivots_before) < 2:
            return None
        swings = find_swings(pivots_before)
        matches = [sw for sw in swings if sw.direction == direction]
        if not matches:
            return None
        return matches[-1]

    def _regime_at(self, i: int) -> str:
        """Régimen de mercado causal (solo datos hasta el día i)."""
        adx_val = self._val("ADX", i)
        plus_di = self._val("PLUS_DI", i)
        minus_di = self._val("MINUS_DI", i)
        close = float(self._features["Close"].iloc[i])
        ema200 = self._val("EMA200", i)
        price_above = (not np.isnan(ema200)) and close > ema200

        date = self._features.index[i]
        highs = self._high_pivots_before(date)
        lows = self._low_pivots_before(date)
        pattern = "indefinido"
        if len(lows) >= 2 and len(highs) >= 2:
            lows_up = lows[-1].price > lows[-2].price
            highs_up = highs[-1].price > highs[-2].price
            if lows_up and highs_up:
                pattern = "ascendente"
            elif (not lows_up) and (not highs_up):
                pattern = "descendente"
            else:
                pattern = "lateral"

        if not np.isnan(adx_val) and adx_val >= 25:
            if pattern == "ascendente" or (price_above and not np.isnan(plus_di) and not np.isnan(minus_di) and plus_di > minus_di):
                return "uptrend"
            if pattern == "descendente" or (not price_above and not np.isnan(plus_di) and not np.isnan(minus_di) and minus_di > plus_di):
                return "downtrend"
            return "mix"
        return "range"

    def evaluate(self, i: int = -1) -> Evaluation:
        """Evalúa la estrategia para un día de la serie.

        Args:
            i: índice posicional (último por defecto). Soportado -1.
        """
        if self._features is None:
            return Evaluation()
        n = len(self._features)
        idx = n + i if i < 0 else i
        if idx < 0 or idx >= n:
            raise IndexError(f"Índice {i} fuera de rango [0, {n})")

        date = self._features.index[idx]

        signal = self._buy_logic(idx)
        exit_needed, exit_reasons = self._exit_logic(idx)
        levels = self._exit_levels(idx)

        # ¿Es una señal de salida FRESCA (nueva) y no un estado crónico?
        exit_fresh = False
        if exit_needed:
            prior = idx - 3
            if prior >= 0:
                was_exit, _ = self._exit_logic(prior)
                exit_fresh = not was_exit
            else:
                exit_fresh = True

        # Indicadores relevantes para reportes
        indicators = {}
        for col in [
            "RSI", "MACD", "MACD_SIGNAL", "ADX", "PLUS_DI", "MINUS_DI",
            "ATR", "STOCH_K", "STOCH_D", "REL_VOL", "EMA20", "EMA50", "EMA200",
        ]:
            indicators[col] = round(self._val(col, idx), 4)

        context = {
            "ticker": self._features.attrs.get("ticker", ""),
            "date": date,
            "price": float(self._features["Close"].iloc[idx]),
            "regime": self._regime_at(idx),
            "exit_reasons": exit_reasons,
        }

        return Evaluation(
            signal=signal,
            should_exit=exit_needed,
            exit_fresh=exit_fresh,
            context=context,
            indicators=indicators,
        )

    def exit_levels_at(self, i: int = -1) -> ExitInfo:
        return self._exit_levels(i)

    # ---- Helpers compartidos de niveles ----

    def _default_exit_levels(
        self, i: int, use_support_sl: bool = True
    ) -> ExitInfo:
        """Armado por defecto de SL/TP según soportes, ATR y Fibonacci."""
        date = self._features.index[i]
        price = float(self._features["Close"].iloc[i])
        atr_val = self._val("ATR", i, default=0.0)

        stop = None
        support = self._nearest_sr_below(price, "support", date)
        if use_support_sl and support is not None:
            stop = support.price - 0.5 * atr_val if atr_val > 0 else support.price
        else:
            low_pivots = self._low_pivots_before(date)
            if low_pivots:
                stop = low_pivots[-1].price - atr_val if atr_val > 0 else low_pivots[-1].price

        # Take-profit: extensión Fibonacci del último swing up
        tp = None
        swing_up = self._last_swing_for("up", date)
        if swing_up is not None:
            fib = compute_fibonacci(swing_up)
            tp = fib.extension_price(self.risk.tp_fib_ratio)
            if tp is not None and not np.isfinite(tp):
                tp = None

        if stop is None or stop >= price:
            stop = price - (2 * atr_val if atr_val > 0 else price * 0.02)
        if stop <= 0:
            stop = price * 0.9

        return ExitInfo(
            stop_loss=stop,
            take_profit=tp,
            trailing_mult=self.risk.trailing_atr_multiplier,
            use_trailing=True,
        )