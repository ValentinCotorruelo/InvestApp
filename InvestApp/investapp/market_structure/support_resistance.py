"""Soportes y resistencias dinámicas desde clústeres de pivotes.

Los niveles se forman agrupando pivotes cercanos y se ponderan según:
- Cantidad de touches (pivotes cerca del nivel).
- Amplitud del clúster.

`build_levels()` es causal: recibe solo los pivotes *ya confirmados* hasta la
fecha de referencia, para que las decisiones nunca vean el futuro.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pivots import Pivot, find_pivots


@dataclass
class SRL:
    """Un nivel de soporte o resistencia detectado."""

    price: float
    kind: str  # "support" | "resistance"
    touches: int
    strength: float  # 0..1 ponderado
    last_touch: pd.Timestamp | None


def _cluster_pivots(pivots: list[Pivot], atr_value: float) -> list[list[Pivot]]:
    """Agrupa pivotes en clústeres de niveles según tolerancia (fracción de ATR)."""
    if not pivots:
        return []
    tolerance = max(atr_value, 1e-6)
    pivots_sorted = sorted(pivots, key=lambda p: p.price)
    clusters: list[list[Pivot]] = []
    for p in pivots_sorted:
        placed = False
        for cluster in clusters:
            ref_price = cluster[0].price
            if abs(p.price - ref_price) <= tolerance * 0.75:
                cluster.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters


def _strength(cluster: list[Pivot], touches: int, recent) -> float:
    """Ponderación simple: más touches y más rango consistente → más fuerte."""
    score = 0.0
    score += min(touches, 5) * 0.15
    prices = [p.price for p in cluster]
    spread = max(prices) / min(prices) - 1 if min(prices) > 0 else 0
    score += (1 / (1 + spread * 100)) * 0.2
    return min(score, 1.0)


def _make_level(cluster: list[Pivot], kind: str, ref_date) -> SRL:
    price = float(np.mean([p.price for p in cluster]))
    return SRL(
        price=price,
        kind=kind,
        touches=len(cluster),
        strength=_strength(cluster, len(cluster), ref_date),
        last_touch=cluster[-1].index,
    )


def build_levels(
    pivots: list[Pivot],
    atr_value: float,
    min_touches: int = 2,
    max_levels: int = 6,
    ref_date=None,
) -> list[SRL]:
    """Arma niveles S/R a partir de una lista de pivotes (ya confirmados).

    Causal: solo usa los pivotes dados (los que eran conocibles hasta la fecha
    de referencia). El precio del nivel es el promedio de ese clúster.
    """
    if not pivots:
        return []

    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]

    res_clusters = _cluster_pivots(highs, atr_value)
    sup_clusters = _cluster_pivots(lows, atr_value)

    levels: list[SRL] = []

    for cluster in res_clusters:
        if len(cluster) >= min_touches:
            levels.append(_make_level(cluster, "resistance", ref_date))

    for cluster in sup_clusters:
        if len(cluster) >= min_touches:
            levels.append(_make_level(cluster, "support", ref_date))

    levels.sort(key=lambda l: l.strength, reverse=True)
    return levels[:max_levels]


def detect_support_resistance(
    df: pd.DataFrame,
    atr_series: pd.Series,
    left: int = 5,
    right: int = 5,
    min_touches: int = 2,
    max_levels: int = 6,
) -> list[SRL]:
    """Detecta niveles de soporte/resistencia sobre el df completo.

    (Modo "foto": usa todo el histórico. Para uso causal día a día preferí
    `build_levels` con pivotes ya confirmados.)
    """
    pivots = find_pivots(df, left=left, right=right)
    last_atr = float(atr_series.dropna().iloc[-1] if atr_series.notna().any() else 0.0)
    return build_levels(pivots, last_atr, min_touches, max_levels)


def nearest_support(prices: pd.Series, levels: list[SRL]) -> SRL | None:
    """Nivel de soporte más cercano por debajo del último precio."""
    price = float(prices.iloc[-1])
    supports = [l for l in levels if l.kind == "support" and l.price < price]
    if not supports:
        return None
    return min(supports, key=lambda l: price - l.price)


def nearest_resistance(prices: pd.Series, levels: list[SRL]) -> SRL | None:
    """Nivel de resistencia más cercano por encima del último precio."""
    price = float(prices.iloc[-1])
    resistances = [l for l in levels if l.kind == "resistance" and l.price > price]
    if not resistances:
        return None
    return min(resistances, key=lambda l: l.price - price)


def nearest_level(price: float, levels: list[SRL]) -> SRL | None:
    """Nivel (S o R) más cercano en distancia absoluta a un precio dado."""
    if not levels:
        return None
    return min(levels, key=lambda l: abs(l.price - price))