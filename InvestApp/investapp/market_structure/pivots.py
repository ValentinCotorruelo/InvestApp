"""Detección de máximos y mínimos pivote (zigzag) y oscilaciones (swings)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    """Un punto pivote (máximo o mínimo) en el tiempo."""

    index: pd.Timestamp
    price: float
    kind: str  # "high" | "low"
    confirmed_at: pd.Timestamp | None = None  # fecha desde la que es "conocible"


@dataclass
class Swing:
    """Una oscilación de precio conectando un pivote bajo con uno alto (o viceversa)."""

    start_index: pd.Timestamp
    start_price: float
    end_index: pd.Timestamp
    end_price: float
    direction: str  # "up" | "down"

    @property
    def magnitude(self) -> float:
        return abs(self.end_price - self.start_price)


def find_pivots(
    df: pd.DataFrame, left: int = 5, right: int = 5, min_move_pct: float = 0.0
) -> list[Pivot]:
    """Encuentra pivotes de máximo/mínimo en una ventana simétrica.

    - Un pivot High es una barra cuyo High es el máximo de [i-left, i+right].
    - Un pivot Low es una barra cuyo Low es el mínimo de [i-left, i+right].
    - Si min_move_pct > 0, filtra pivotes con movimiento mínimo relativo
      (zigzag simplificado por filtrado posterior).
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    index = df.index
    n = len(df)

    pivots: list[Pivot] = []
    for i in range(left, n - right):
        is_high = (
            high[i] == max(high[i - left : i + right + 1])
            and low[i] != min(low[i - left : i + right + 1])
        )
        is_low = (
            low[i] == min(low[i - left : i + right + 1])
            and high[i] != max(high[i - left : i + right + 1])
        )
        # Evita pivotes en zonas sin movimiento (rango horizontal)
        window_range = max(high[i - left : i + right + 1]) - min(
            low[i - left : i + right + 1]
        )
        if window_range <= 0:
            continue

        if is_high:
            pivots.append(Pivot(index=index[i], price=float(high[i]), kind="high",
                                confirmed_at=index[min(i + right, n - 1)]))
        elif is_low:
            pivots.append(Pivot(index=index[i], price=float(low[i]), kind="low",
                                confirmed_at=index[min(i + right, n - 1)]))

    if min_move_pct > 0:
        pivots = _filter_min_move(pivots, min_move_pct)
    return pivots


def _filter_min_move(pivots: list[Pivot], min_move_pct: float) -> list[Pivot]:
    """Zigzag: elimina pivotes cuyo movimiento respecto al previo es menor al umbral."""
    if len(pivots) < 2:
        return pivots
    filtered: list[Pivot] = [pivots[0]]
    for p in pivots[1:]:
        last = filtered[-1]
        move = abs(p.price - last.price) / last.price
        if move >= min_move_pct:
            filtered.append(p)
        else:
            # Reemplazar el pivote previo del mismo tipo si el movimiento es mayor
            if len(filtered) >= 2:
                prev = filtered[-2]
                if p.kind == prev.kind:
                    filtered[-1] = p
    return filtered


def find_swings(pivots: list[Pivot]) -> list[Swing]:
    """Construye oscilaciones alternando high/low pivotes."""
    if len(pivots) < 2:
        return []
    swings: list[Swing] = []
    for i in range(len(pivots) - 1):
        a, b = pivots[i], pivots[i + 1]
        direction = "up" if b.price > a.price else "down"
        swings.append(
            Swing(
                start_index=a.index,
                start_price=a.price,
                end_index=b.index,
                end_price=b.price,
                direction=direction,
            )
        )
    return swings


def last_swing_low(pivots: list[Pivot]) -> Pivot | None:
    """Último pivote mínimo (relevante para stops)."""
    lows = [p for p in pivots if p.kind == "low"]
    return lows[-1] if lows else None


def last_swing_high(pivots: list[Pivot]) -> Pivot | None:
    """Último pivote máximo (relevante para resistencias/take-profit)."""
    highs = [p for p in pivots if p.kind == "high"]
    return highs[-1] if highs else None


def recent_pivots(pivots: list[Pivot], n: int = 6) -> list[Pivot]:
    """Devuelve los últimos 'n' pivotes ordenados cronológicamente."""
    return pivots[-n:] if len(pivots) <= n else pivots[-n:]