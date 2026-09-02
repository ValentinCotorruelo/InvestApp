"""Estrategia Trend Follower: sigue tendencias fuertes con confluencia
de medias, MACD, ADX, ruptura de resistencia y volumen.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..market_structure.fibonacci import compute_fibonacci
from .base import ExitInfo, Signal, Strategy


class TrendFollowerStrategy(Strategy):
    name = "trend_follower"
    description = (
        "Sigue tendencias fuertes: EMA200 alcista, ADX>25, cruce EMA20/50, "
        "MACD alcista, ruptura de resistencia con volumen. "
        "SL por soporte/ATR, TP por extensión Fibonacci."
    )

    ADX_MIN = 25.0
    SL_ATR_EXTRAS = 1.0

    def _is_uptrend_context(self, i: int) -> bool:
        date = self._features.index[i]
        close = float(self._features["Close"].iloc[i])

        ema200 = self._val("EMA200", i)
        if np.isnan(ema200) or close <= ema200:
            return False

        adx_val = self._val("ADX", i)
        plus_di = self._val("PLUS_DI", i)
        minus_di = self._val("MINUS_DI", i)
        if np.isnan(adx_val) or adx_val < self.ADX_MIN:
            return False
        if np.isnan(plus_di) or np.isnan(minus_di) or plus_di <= minus_di:
            return False

        # Estructura ascendente: mínimos pivote crecientes
        lows = self._low_pivots_before(date)
        if len(lows) >= 2 and lows[-1].price <= lows[-2].price:
            return False

        return True

    def _broken_resistance(self, i: int) -> Optional[float]:
        """Si el precio rompió una resistencia previa (cierre previo bajo el
        nivel, cierre actual sobre el nivel), devuelve el precio del nivel."""
        date = self._features.index[i]
        close = float(self._features["Close"].iloc[i])
        prev_close = float(self._features["Close"].iloc[i - 1])
        atr_val = self._val("ATR", i, default=0.0)
        tol = 1.5 * atr_val if atr_val > 0 else close * 0.01

        candidates = [
            lvl
            for lvl in self._sr_levels_at(date)
            if lvl.kind == "resistance"
            and lvl.price > prev_close
            and lvl.price <= close + tol
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda lvl: lvl.price).price

    def _buy_logic(self, i: int) -> Optional[Signal]:
        if i < 60:
            return None
        f = self._features

        # 1. Contexto
        if not self._is_uptrend_context(i):
            return None

        # 2. Estructura de medias: EMA20 sobre EMA50 (golden cross ya establecido
        #    o producido recientemente)
        e20 = self._val("EMA20", i)
        e50 = self._val("EMA50", i)
        if np.isnan(e20) or np.isnan(e50) or e20 <= e50:
            return None
        fresh_cross = self._any_within(
            lambda j: self._crossed_above("EMA20", "EMA50", j), i, 5
        )

        # 3. MACD por encima de señal y de cero
        macd_val = self._val("MACD", i)
        macd_sig = self._val("MACD_SIGNAL", i)
        if np.isnan(macd_val) or np.isnan(macd_sig):
            return None
        if not (macd_val > macd_sig and macd_val > 0):
            return None

        # 4. Ruptura de resistencia reciente O cruce reciente de medias
        snapped = self._broken_resistance(i)
        breakout_recent = self._any_within(
            lambda j: self._broken_resistance(j) is not None, i, 3
        )
        if not (fresh_cross or breakout_recent):
            return None

        # 5. Volumen de confirmación
        rel_vol = self._val("REL_VOL", i, default=0.0)
        if np.isnan(rel_vol) or rel_vol < 1.0:
            return None

        price = float(f["Close"].iloc[i])
        levels = self._exit_levels(i)
        reasons = [
            "Precio sobre EMA200",
            f"ADX={self._val('ADX', i):.1f} con +DI>-DI",
            "EMA20 sobre EMA50" + (" (cruce reciente)" if fresh_cross else ""),
            "MACD sobre señal y sobre cero",
            (f"Ruptura de resistencia {snapped:.2f}" if snapped is not None else "Cruce de medias reciente"),
            f"Volumen relativo {rel_vol:.2f}",
        ]
        strength = min(1.0, 0.5 + 0.1 * (rel_vol - 1.0))

        return Signal(
            side="long",
            entry=price,
            stop_loss=levels.stop_loss or 0.0,
            take_profit=levels.take_profit,
            reasons=reasons,
            strength=strength,
            meta={"breakout_resistance": snapped},
        )

    def _exit_logic(self, i: int) -> tuple[bool, list[str]]:
        if i < 60:
            return False, []

        # Salida por momentum: MACD cruza bajo señal y precio bajo EMA20
        if self._crossed_below("MACD", "MACD_SIGNAL", i):
            close = float(self._features["Close"].iloc[i])
            ema20 = self._val("EMA20", i)
            if not np.isnan(ema20) and close < ema20:
                return True, ["MACD bajo señal y precio bajo EMA20"]

        # Salida por estructura: máximo pivote más bajo que el anterior
        date = self._features.index[i]
        highs = self._high_pivots_before(date)
        if len(highs) >= 2 and highs[-1].price <= highs[-2].price:
            # solo considerar si perdimos al menos un poco de momentum (ADX cae)
            adx_val = self._val("ADX", i)
            if not np.isnan(adx_val) and adx_val < 30:
                return True, ["Máximo pivote descendente (reversión de estructura)"]

        # Salida por tendencia: ADX < 20 o cierre bajo EMA50
        adx_val = self._val("ADX", i)
        if not np.isnan(adx_val) and adx_val < 20:
            return True, ["ADX < 20 (pérdida de tendencia)"]

        close = float(self._features["Close"].iloc[i])
        ema50 = self._val("EMA50", i)
        if not np.isnan(ema50) and close < ema50:
            return True, ["Cierre bajo EMA50"]

        return False, []

    def _exit_levels(self, i: int) -> ExitInfo:
        return self._default_exit_levels(i, use_support_sl=True)