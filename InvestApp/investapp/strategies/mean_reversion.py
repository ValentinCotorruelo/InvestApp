"""Estrategia Mean Reversion: compra en rangos laterales cuando el precio
toca el extremo inferior (Bollinger, RSI, estocástico) y hay soporte
o retracement de Fibonacci en la zona.
"""
from __future__ import annotations

import numpy as np

from ..market_structure.fibonacci import compute_fibonacci
from .base import ExitInfo, Signal, Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    description = (
        "Regresa a la media en rangos laterales: ADX<20, cierre bajo banda "
        "inferior de Bollinger, RSI<30, estocástico sobrevendido con giro, "
        "apoyado en soporte o retracement Fibonacci 0.618/0.786."
    )

    ADX_MAX = 20.0
    RSI_MAX = 40.0

    def _is_range_context(self, i: int) -> bool:
        adx_val = self._val("ADX", i)
        if np.isnan(adx_val) or adx_val > self.ADX_MAX:
            return False
        # Precio dentro de bandas (no tendencia rota)
        close = float(self._features["Close"].iloc[i])
        bb_u = self._val("BB_UPPER", i)
        bb_l = self._val("BB_LOWER", i)
        if np.isnan(bb_u) or np.isnan(bb_l):
            return False
        return bb_l < close < bb_u

    def _on_support_or_retracement(self, i: int) -> float:
        """Devuelve el nivel clave si el precio está en soporte o retracement fib."""
        date = self._features.index[i]
        price = float(self._features["Close"].iloc[i])

        # Soporte cercano (dentro de 1.2 * ATR)
        atr_val = self._val("ATR", i, default=0.0)
        tol = 1.2 * atr_val if atr_val > 0 else price * 0.02
        support = self._nearest_sr_below(price, "support", date)
        if support is not None and (price - support.price) <= tol:
            return support.price

        # Retracement Fibonacci en zona dorada / profunda
        swing_up = self._last_swing_for("up", date)
        if swing_up is not None:
            fib = compute_fibonacci(swing_up)
            for ratio in (0.618, 0.786):
                level = fib.retracement_price(ratio)
                if level is not None and abs(price - level) <= tol:
                    return level

        return 0.0

    def _buy_logic(self, i: int) -> Signal | None:
        if i < 60:
            return None
        f = self._features

        # 1. Contexto lateral (vigente)
        if not self._is_range_context(i):
            return None

        # 2. Tocó banda inferior recientemente
        close = float(f["Close"].iloc[i])
        bb_lower = self._val("BB_LOWER", i)
        if np.isnan(bb_lower):
            return None
        touched_band = close < bb_lower or self._any_within(
            lambda j: self._val("BB_LOWER", j) is not None
            and float(f["Close"].iloc[j]) < self._val("BB_LOWER", j),
            i - 1, 2,
        )
        if not touched_band:
            return None

        # 3. RSI sobrevendido
        rsi_val = self._val("RSI", i)
        if np.isnan(rsi_val) or rsi_val >= self.RSI_MAX:
            return None

        # 4. Estocástico sobrevendido CON giro alcista (reciente en 3 días)
        k = self._val("STOCH_K", i)
        d = self._val("STOCH_D", i)
        if np.isnan(k) or np.isnan(d) or k >= 30:
            return None
        k_d_turn = self._any_within(
            lambda j: self._stoch_turn_up(j), i, 3
        )
        if not k_d_turn:
            return None

        # 5. Soporte o retracement fib (reciente en 3 días)
        key_level = self._on_support_or_retracement(i)
        if key_level <= 0:
            key_level = self._recent_key_level(i)
            if key_level <= 0:
                return None

        # 6. Confirmación de volumen o vela de reversión
        rel_vol = self._val("REL_VOL", i, default=0.0)
        row = f.iloc[i]
        rng = row["High"] - row["Low"]
        close_pos = (close - row["Low"]) / rng if rng > 0 else 0.5
        body_up = close > row["Open"]
        candle_rev = body_up and close_pos >= 0.6
        if not (rel_vol > 0.9 or candle_rev):
            return None

        levels = self._exit_levels(i)
        reasons = [
            "ADX<20 (mercado lateral)",
            f"Toca banda inferior ({bb_lower:.2f})",
            f"RSI={rsi_val:.1f} (<32 sobrevendido)",
            f"Estocástico %K={k:.1f} con giro alcista",
            f"Soporte/Fibonacci en {key_level:.2f}",
            "Confirmación de volumen/vela de reversión",
        ]
        strength = 0.6 + 0.4 * min(1.0, (self.RSI_MAX - rsi_val) / self.RSI_MAX)

        return Signal(
            side="long",
            entry=close,
            stop_loss=levels.stop_loss or 0.0,
            take_profit=levels.take_profit,
            reasons=reasons,
            strength=min(1.0, strength),
            meta={"key_level": key_level},
        )

    def _stoch_turn_up(self, j: int) -> bool:
        """%K < 30 y giró alcista (cruza sobre %D) en el día j."""
        if j < 1:
            return False
        f = self._features
        k = self._val("STOCH_K", j)
        d = self._val("STOCH_D", j)
        if np.isnan(k) or np.isnan(d) or k >= 30:
            return False
        k_arr = f["STOCH_K"].to_numpy()
        d_arr = f["STOCH_D"].to_numpy()
        return k_arr[j - 1] <= d_arr[j - 1] and k > d

    def _recent_key_level(self, i: int) -> float:
        """Nivel clave en los 2 días previos (si el rebote ocurrió antes de hoy)."""
        if i < 1:
            return 0.0
        price = float(self._features["Close"].iloc[i])
        atr_val = self._val("ATR", i, default=0.0)
        tol = 1.2 * atr_val if atr_val > 0 else price * 0.02
        for j in range(max(0, i - 2), i):
            prior = self._on_support_or_retracement(j)
            if prior > 0 and abs(price - prior) <= 2 * tol:
                return prior
        return 0.0

    def _exit_logic(self, i: int) -> tuple[bool, list[str]]:
        if i < 60:
            return False, []

        # Salida por momentum: RSI > 60 y estocástico > 80
        rsi_val = self._val("RSI", i)
        k = self._val("STOCH_K", i)
        if not np.isnan(rsi_val) and rsi_val > 60 and not np.isnan(k) and k > 80:
            return True, ["RSI>60 y estocástico>80 (sobrecompra)"]

        # Salida por contexto: el rango se volvió tendencia
        adx_val = self._val("ADX", i)
        if not np.isnan(adx_val) and adx_val >= 25:
            return True, ["ADX>=25 (el rango se convirtió en tendencia)"]

        # El precio cerró por encima de banda media = regresión completada
        close = float(self._features["Close"].iloc[i])
        bb_mid = self._val("BB_MID", i)
        bb_upper = self._val("BB_UPPER", i)
        if not np.isnan(bb_upper) and close >= bb_upper:
            return True, ["Cierre en banda superior (objetivo cumplido)"]
        if not np.isnan(bb_mid) and close > bb_mid:
            return True, ["Regresión a banda media completada"]

        return False, []

    def _exit_levels(self, i: int) -> ExitInfo:
        info = self._default_exit_levels(i, use_support_sl=True)
        info.trailing_mult = 2.0  # trailing más ajustado en mean reversion
        return info