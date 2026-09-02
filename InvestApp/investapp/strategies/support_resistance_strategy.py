"""Estrategia de Soportes y Resistencias (pura, sin otros indicadores).

Señales basadas exclusivamente en niveles de precio detectados por clústeres
de pivotes. Dos tipos de entrada:

  - **Rebote en soporte**: el precio toca un soporte y cierra por encima
    (confirmación de rebote).
  - **Ruptura de resistencia**: el precio cierra por encima de una resistencia
    con margen suficiente (breakout confirmado).

Las salidas se basan en la pérdida del soporte o el alcance de la resistencia
objetivo. Sin RSI, sin MACD, sin EMA — solo precio y niveles.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ExitInfo, Signal, Strategy

# ---- Parámetros (todos configurables) ----
PIVOT_LEFT = 5
PIVOT_RIGHT = 3             # reducido: pivote confirmado en 3 días (no 5)
CLUSTER_TOLERANCE_MULT = 2.0  # ATR × 2.0 para agrupar pivotes cercanos
MIN_TOUCHES = 2             # mínimo de toques por zona S/R
BOUNCE_MARGIN_MULT = 0.5    # ATR × 0.5: margen para "tocar" un nivel
BREAKOUT_PCT = 0.01         # 1%: margen mínimo para confirmar ruptura
MIN_RR = 1.5                # R/R mínimo para aceptar la señal
MAX_LOOKBACK = 750          # barras históricas máximas


class SupportResistanceStrategy(Strategy):
    name = "support_resistance"
    description = (
        "Pura S/R: compra por rebote en soporte o ruptura de resistencia. "
        "Sin otros indicadores — solo precio y niveles."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, min_bars_required=60, **kwargs)

    def _buy_logic(self, i: int) -> Optional[Signal]:
        if i < self.min_bars_required:
            return None

        date = self._features.index[i]
        price = float(self._features["Close"].iloc[i])
        prev_close = float(self._features["Close"].iloc[i - 1]) if i > 0 else price
        atr_val = self._val("ATR", i, default=0.0)
        if atr_val <= 0:
            return None

        levels = self._sr_levels_at(date)
        supports = [lvl for lvl in levels if lvl.kind == "support" and lvl.price < price]
        resistances = [lvl for lvl in levels if lvl.kind == "resistance" and lvl.price > price]

        # ---- Condición A: Rebote en soporte ----
        bounce_margin = BOUNCE_MARGIN_MULT * atr_val
        best_support = None
        for s in supports:
            if price <= s.price + bounce_margin:
                if best_support is None or s.price > best_support.price:
                    best_support = s

        if best_support is not None and price > best_support.price:
            # El precio tocó el soporte (dentro del margen) y cerró por encima
            sl = best_support.price - bounce_margin
            tp = self._find_target_resistance(best_support.price, price, resistances, atr_val)
            risk = price - sl
            reward = tp - price if tp is not None else 0.0
            if risk > 0 and reward / risk >= MIN_RR:
                reasons = [
                    f"Rebote en soporte {best_support.price:.2f} "
                    f"(toques={best_support.touches})",
                    f"Close={price:.2f} > soporte",
                    f"R/R={reward / risk:.2f}",
                ]
                return Signal(
                    side="long",
                    entry=price,
                    stop_loss=sl,
                    take_profit=tp,
                    reasons=reasons,
                    strength=min(1.0, 0.5 + 0.1 * best_support.touches),
                    meta={
                        "type": "bounce",
                        "support": best_support.price,
                        "rr": reward / risk if risk > 0 else 0.0,
                    },
                )

        # ---- Condición B: Ruptura de resistencia ----
        # La resistencia ROI queda POR DEBAJO del precio actual (ya superada).
        rupturas = [
            lvl for lvl in levels
            if lvl.kind == "resistance" and lvl.price < price
        ]
        for r in rupturas:
            if prev_close <= r.price and price > r.price * (1 + BREAKOUT_PCT):
                sl = r.price - bounce_margin
                tp = self._find_target_resistance(
                    r.price, price, resistances, atr_val
                )
                risk = price - sl
                reward = tp - price if tp is not None else 0.0
                if risk > 0 and reward / risk >= MIN_RR:
                    reasons = [
                        f"Ruptura de resistencia {r.price:.2f} "
                        f"(toques={r.touches})",
                        f"Prev close={prev_close:.2f} <= {r.price:.2f} < Close={price:.2f}",
                        f"R/R={reward / risk:.2f}",
                    ]
                    return Signal(
                        side="long",
                        entry=price,
                        stop_loss=sl,
                        take_profit=tp,
                        reasons=reasons,
                        strength=min(1.0, 0.5 + 0.1 * r.touches),
                        meta={
                            "type": "breakout",
                            "resistance": r.price,
                            "rr": reward / risk if risk > 0 else 0.0,
                        },
                    )

        return None

    def _exit_logic(self, i: int) -> tuple[bool, list[str]]:
        if i < self.min_bars_required:
            return False, []

        date = self._features.index[i]
        price = float(self._features["Close"].iloc[i])
        atr_val = self._val("ATR", i, default=0.0)
        levels = self._sr_levels_at(date)

        # El soporte más cercano por debajo del precio
        supports = [lvl for lvl in levels if lvl.kind == "support" and lvl.price < price]
        nearest_support = max(supports, key=lambda x: x.price) if supports else None

        # 1) Salida: el precio CIERRA por debajo del soporte (pérdida del nivel)
        if nearest_support is not None and price < nearest_support.price:
            return True, [f"Cierre bajo soporte {nearest_support.price:.2f}"]

        # 2) Salida: el precio cerró dentro del margen bajo la resistencia
        #    (objetivo prácticamente cumplido).
        resistances = [lvl for lvl in levels if lvl.kind == "resistance" and lvl.price > price]
        if resistances and atr_val > 0:
            nearest_res = min(resistances, key=lambda x: x.price)
            if price >= nearest_res.price - 0.2 * atr_val:
                return True, [f"Cerca del objetivo en resistencia {nearest_res.price:.2f}"]

        return False, []

    def _exit_levels(self, i: int) -> ExitInfo:
        return self._default_exit_levels(i, use_support_sl=True)

    def _find_target_resistance(
        self,
        from_price: float,
        current_price: float,
        resistances: list,
        atr_val: float,
    ) -> Optional[float]:
        """Busca la resistencia más cercana por encima como target TP."""
        candidates = [r for r in resistances if r.price > current_price]
        if candidates:
            target = min(r.price for r in candidates)
            return target
        # Si no hay resistencia, usar objetivo implícito: +15% del nivel base
        return current_price * 1.15
