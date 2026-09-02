"""Retrocesos y extensiones de Fibonacci basados en oscilaciones (swings).

- Retrocesos: 0.236, 0.382, 0.5, 0.618, 0.786
- Extensiones: 1.272, 1.618, 2.618, 3.618
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pivots import Swing


RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
EXTENSION_LEVELS = [0.0, 1.272, 1.618, 2.618, 3.618]


@dataclass
class FibonacciLevel:
    """Un nivel de Fibonacci derivado de un swing."""

    swing: Swing
    ratio: float
    price: float
    kind: str  # "retracement" | "extension"


@dataclass
class FibonacciResult:
    """Retrocesos y extensiones de un swing dado."""

    swing: Swing
    retracements: list[FibonacciLevel] = field(default_factory=list)
    extensions: list[FibonacciLevel] = field(default_factory=list)

    def retracement_price(self, ratio: float) -> float | None:
        for lv in self.retracements:
            if abs(lv.ratio - ratio) < 1e-6:
                return lv.price
        return None

    def extension_price(self, ratio: float) -> float | None:
        for lv in self.extensions:
            if abs(lv.ratio - ratio) < 1e-6:
                return lv.price
        return None


def _wider_pool(swing: Swing) -> float:
    """Punto de origen más lejano del swing (para números de Fibonacci)."""
    if swing.direction == "up":
        return swing.start_price
    return swing.start_price


def compute_fibonacci(swing: Swing) -> FibonacciResult:
    """Calcula retrocesos y extensiones para un swing.

    Swing "up" (bajo→alto): el rango es start_price→end_price ascendente.
    Swing "down" (alto→bajo): el rango es start_price→end_price descendente.
    """
    high = max(swing.start_price, swing.end_price)
    low = min(swing.start_price, swing.end_price)
    if high - low <= 0:
        # Swing plano: sin niveles
        return FibonacciResult(swing=swing)

    result = FibonacciResult(swing=swing)

    # Retrocesos: retroceden desde el extremo del movimiento hacia el origen.
    # Si el swing sube, los retrocesos van hacia abajo desde "high" hasta "low".
    # Si baja, van hacia arriba desde "low" hasta "high".
    for ratio in RETRACEMENT_LEVELS:
        if swing.direction == "up":
            price = high - ratio * (high - low)
        else:
            price = low + ratio * (high - low)
        result.retracements.append(
            FibonacciLevel(swing=swing, ratio=ratio, price=price, kind="retracement")
        )

    # Extensiones: proyecciones más allá del extremo del movimiento.
    extent = high - low
    for ratio in EXTENSION_LEVELS:
        if swing.direction == "up":
            price = low + extent * (1.0 + ratio)
        else:
            price = high - extent * (1.0 + ratio)
        result.extensions.append(
            FibonacciLevel(swing=swing, ratio=ratio, price=price, kind="extension")
        )

    return result


def golden_zone(fib: FibonacciResult) -> tuple[float, float] | None:
    """Zona dorada: entre retracement 0.5 y 0.618 del swing vigente."""
    if not fib.retracements:
        return None
    r_50 = fib.retracement_price(0.5)
    r_618 = fib.retracement_price(0.618)
    if r_50 is None or r_618 is None:
        return None
    lo = min(r_50, r_618)
    hi = max(r_50, r_618)
    return (lo, hi)


def in_golden_zone(price: float, fib: FibonacciResult, tol_pct: float = 0.02) -> bool:
    """¿El precio está dentro de la zona dorada (0.5-0.618)? Con tolerancia extra."""
    zone = golden_zone(fib)
    if zone is None:
        return False
    lo, hi = zone
    tol = (hi - lo) * tol_pct
    return (lo - tol) <= price <= (hi + tol)