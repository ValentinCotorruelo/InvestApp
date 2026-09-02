"""Estrategia Hybrid: selecciona dinámicamente la sub-estrategia según
el régimen de mercado detectado (ADX + estructura).

- ADX > 25 con dirección → Trend Follower
- ADX < 20 → Mean Reversion
- 20-25 (zona gris) → Confluence
"""
from __future__ import annotations

import numpy as np

from .base import Evaluation, ExitInfo, Signal, Strategy
from .confluence import ConfluenceStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_follower import TrendFollowerStrategy

TREND_ADX_MIN = 25.0
RANGE_ADX_MAX = 20.0
COOLDOWN_AFTER_STOP = 5  # días tras un stop-loss (lo aplica el engine vía meta)


class HybridStrategy(Strategy):
    name = "hybrid"
    description = (
        "Modo dinámico: Trend Follower si ADX>25 con dirección, Mean Reversion "
        "si ADX<20, Confluence en zona gris (20-25)."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trend_strat = TrendFollowerStrategy(risk=self.risk)
        self._mr_strat = MeanReversionStrategy(risk=self.risk)
        self._conf_strat = ConfluenceStrategy(risk=self.risk)

    @property
    def mode(self) -> str:
        """Modo actual según último valor de ADX/contexto (requiere prepare)."""
        if self._trend is None:
            return "unknown"
        if self._trend.regime == "uptrend" or self._trend.regime == "downtrend":
            return "trend"
        if self._trend.regime == "range":
            return "mean_reversion"
        return "confluence"

    def prepare(self, df):
        super().prepare(df)
        # Preparamos las sub-estrategias sobre el mismo df
        self._trend_strat.prepare(df)
        self._mr_strat.prepare(df)
        self._conf_strat.prepare(df)
        return self._features

    def _select_sub(self, i: int) -> Strategy:
        adx_val = self._val("ADX", i)
        if np.isnan(adx_val):
            return self._conf_strat
        if adx_val >= TREND_ADX_MIN:
            return self._trend_strat
        if adx_val < RANGE_ADX_MAX:
            return self._mr_strat
        return self._conf_strat

    def _buy_logic(self, i: int) -> Signal | None:
        sub = self._select_sub(i)
        return sub._buy_logic(i)

    def _exit_logic(self, i: int) -> tuple[bool, list[str]]:
        sub = self._select_sub(i)
        return sub._exit_logic(i)

    def _exit_levels(self, i: int) -> ExitInfo:
        sub = self._select_sub(i)
        return sub._exit_levels(i)

    def _mode_name(self, i: int) -> str:
        """Nombre del modo para el día i (causal, sin mirar el futuro)."""
        sub = self._select_sub(i)
        return sub.name

    def evaluate(self, i: int = -1) -> Evaluation:
        ev = super().evaluate(i)
        n = len(self._features)
        idx = n + i if i < 0 else i
        ev.context["mode"] = self._mode_name(idx)
        return ev