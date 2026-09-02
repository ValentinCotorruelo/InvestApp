"""Gestión de riesgo: stops, take-profit y trailing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExitLevels:
    """Niveles de salida predefinidos para una posición."""

    stop_loss: float
    take_profit: float | None = None
    trailing_stop_mult: float | None = None  # multiplicador de ATR
    trailing_stop: float | None = None  # valor absoluto del trailing actual

    def __post_init__(self):
        if self.stop_loss <= 0:
            raise ValueError("stop_loss debe ser positivo")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit debe ser positivo")
        if self.take_profit is not None and self.take_profit <= self.stop_loss:
            raise ValueError("take_profit debe ser mayor que stop_loss")


@dataclass
class RiskManagementParams:
    """Parámetros globales de gestión de riesgo.

    atr_period: período del ATR.
    sl_atr_multiplier: stop-loss como múltiplo de ATR (ej. 2).
    tp_fib_ratio: ratio de extensión Fibonacci para take-profit.
    trailing_atr_multiplier: múltiplo de ATR para el trailing stop.
    min_rr: ratio R/R mínimo para abrir una operación.
    cooldown_days: días de espera tras un stop-loss.
    """

    atr_period: int = 14
    sl_atr_multiplier: float = 2.0
    tp_fib_ratio: float = 1.618
    trailing_atr_multiplier: float = 3.0
    min_rr: float = 1.5
    cooldown_days: int = 5


def initial_stop_loss_atr(entry_atr: float, multiplier: float = 2.0) -> float:
    """Stop-loss basado en ATR (sube desde el precio)."""
    return entry_atr * multiplier


def update_trailing_stop(
    current_trailing: float | None,
    current_price: float,
    atr_value: float,
    multiplier: float,
) -> float:
    """Actualiza un trailing stop de compra (sigue máximos)."""
    new = current_price - atr_value * multiplier
    if current_trailing is None:
        return max(0.0, new)
    return max(current_trailing, new)


def should_stop_out(current_price: float, stop_price: float, side: str = "long") -> bool:
    """¿El precio tocó el stop?"""
    if side == "long":
        return current_price <= stop_price
    return current_price >= stop_price


def should_take_profit(current_price: float, target: float, side: str = "long") -> bool:
    """¿El precio tocó el take-profit?"""
    if side == "long":
        return current_price >= target
    return current_price <= target