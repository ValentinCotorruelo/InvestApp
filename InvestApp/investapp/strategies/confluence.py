"""Estrategia Confluence: sistema de voto ponderado multi-indicador.

6 factores puntúan a favor del LONG; la operación se habilita cuando la
confluencia supera el umbral y no hay veto activo.
"""
from __future__ import annotations

import numpy as np

from ..market_structure.fibonacci import compute_fibonacci, in_golden_zone
from .base import ExitInfo, Signal, Strategy

# (factor, peso)
FACTORS = [
    ("tendencia_estructural", 2),
    ("rsi_sano", 1),
    ("macd_alcista", 1),
    ("ema_cruce", 1),
    ("soporte_resistencia", 2),
    ("fibonacci_zona_dorada", 2),
]
SCORE_MIN = 6  # de 9 posibles
MAX_SCORE = 9
VOLUME_RATIO_MIN = 1.0
MIN_RR = 1.5


class ConfluenceStrategy(Strategy):
    name = "confluence"
    description = (
        "Compra solo con alta confluencia (≥7/9): tendencia estructural, RSI sano, "
        "MACD alcista, EMA20>EMA50, soporte/resistencia y zona dorada de Fibonacci, "
        "con volumen y R/R ≥1.5."
    )

    def _factors(self, i: int) -> list[tuple[str, bool]]:
        date = self._features.index[i]
        price = float(self._features["Close"].iloc[i])

        # 1. Tendencia estructural
        ema200 = self._val("EMA200", i)
        lows = self._low_pivots_before(date)
        struct_up = not np.isnan(ema200) and price > ema200 and len(lows) >= 2 and (
            lows[-1].price > lows[-2].price
        )

        # 2. RSI sano
        rsi_val = self._val("RSI", i)
        rsi_ok = not np.isnan(rsi_val) and 50 < rsi_val < 70

        # 3. MACD alcista
        macd_val = self._val("MACD", i)
        macd_sig = self._val("MACD_SIGNAL", i)
        macd_ok = not np.isnan(macd_val) and not np.isnan(macd_sig) and macd_val > macd_sig and macd_val > 0

        # 4. EMA20 > EMA50
        e20 = self._val("EMA20", i)
        e50 = self._val("EMA50", i)
        ema_ok = not np.isnan(e20) and not np.isnan(e50) and e20 > e50

        # 5. Soporte / resistencia (reciente en 2 días)
        atr_val = self._val("ATR", i, default=0.0)
        tol = 1.5 * atr_val if atr_val > 0 else price * 0.02
        support = self._nearest_sr_below(price, "support", date)
        near_support = support is not None and (price - support.price) <= tol
        res = self._nearest_sr_above(price, "resistance", date)
        breakout = res is not None and price > res.price
        if not (near_support or breakout):
            # ventana de 2 días
            for j in (i - 1, i):
                if j < 0:
                    continue
                date_j = self._features.index[j]
                pw = float(self._features["Close"].iloc[j])
                sup_j = self._nearest_sr_below(pw, "support", date_j)
                if sup_j is not None and (pw - sup_j.price) <= tol:
                    near_support = True
                    break
                res_j = self._nearest_sr_above(pw, "resistance", date_j)
                if res_j is not None and pw > res_j.price:
                    breakout = True
                    break
        rel_vol = self._val("REL_VOL", i, default=0.0)
        sr_ok = (near_support or breakout) and not np.isnan(rel_vol) and rel_vol > 1.0

        # 6. Fibonacci zona (amplia: 0.382-0.786, con foco en la dorada)
        fib_ok = False
        swing_up = self._last_swing_for("up", date)
        if swing_up is not None:
            fib = compute_fibonacci(swing_up)
            fib_ok = in_golden_zone(price, fib, tol_pct=0.03)
            if not fib_ok:
                # Zona más amplia 0.382-0.786
                r38 = fib.retracement_price(0.382)
                r786 = fib.retracement_price(0.786)
                if r38 is not None and r786 is not None:
                    lo, hi = min(r38, r786), max(r38, r786)
                    tol_fib = (hi - lo) * 0.05
                    fib_ok = (lo - tol_fib) <= price <= (hi + tol_fib)

        return [
            ("tendencia_estructural", struct_up),
            ("rsi_sano", rsi_ok),
            ("macd_alcista", macd_ok),
            ("ema_cruce", ema_ok),
            ("soporte_resistencia", sr_ok),
            ("fibonacci_zona_dorada", fib_ok),
        ]

    def _score(self, i: int) -> tuple[int, dict[str, bool]]:
        factors = self._factors(i)
        score = 0
        detail = {}
        for name, ok in factors:
            weight = dict(FACTORS)[name]
            if ok:
                score += weight
            detail[name] = ok
        return score, detail

    def _veto_active(self, i: int) -> bool:
        rsi_val = self._val("RSI", i)
        if not np.isnan(rsi_val) and rsi_val < 30:
            return True
        macd_val = self._val("MACD", i)
        if not np.isnan(macd_val) and macd_val < 0 and self._crossed_below("MACD", "MACD_SIGNAL", i):
            return True
        return False

    def _buy_logic(self, i: int) -> Signal | None:
        if i < 60:
            return None
        price = float(self._features["Close"].iloc[i])

        score, detail = self._score(i)
        if score < SCORE_MIN:
            return None
        if self._veto_active(i):
            return None

        # Confirmación de volumen
        rel_vol = self._val("REL_VOL", i, default=0.0)
        if np.isnan(rel_vol) or rel_vol < VOLUME_RATIO_MIN:
            return None

        levels = self._exit_levels(i)
        sl = levels.stop_loss or 0.0
        tp = levels.take_profit
        if sl <= 0:
            return None

        # R/R mínimo
        if tp is not None:
            rr = (tp - price) / (price - sl)
            if rr < MIN_RR:
                return None

        reasons = [f"{name}={'V' if ok else 'X'}" for name, ok in detail.items()]
        reasons.append(f"Score {score}/{MAX_SCORE}")
        reasons.append(f"Volumen relativo {rel_vol:.2f}")

        return Signal(
            side="long",
            entry=price,
            stop_loss=sl,
            take_profit=tp,
            reasons=reasons,
            strength=min(1.0, score / MAX_SCORE),
            meta={"score": score, "detail": detail},
        )

    def _exit_logic(self, i: int) -> tuple[bool, list[str]]:
        if i < 60:
            return False, []
        score, detail = self._score(i)
        # Salimos cuando la confluencia se pierde (~2 factores caen)
        if score <= SCORE_MIN - 2:
            lost = [k for k, v in detail.items() if not v]
            return True, [f"Confluencia perdida ({score}/{MAX_SCORE}): " + ", ".join(lost)]
        if self._veto_active(i):
            return True, ["Veto activo (RSI<35 o MACD negativo cruzando bajo)"]
        return False, []

    def _exit_levels(self, i: int) -> ExitInfo:
        # SL bajo mínimo pivote, TP fib/ resistencia
        return self._default_exit_levels(i, use_support_sl=True)