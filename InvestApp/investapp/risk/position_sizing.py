"""Tamaño de posición basado en riesgo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PositionSizingParams:
    """Parámetros de tamaño de posición.

    risk_per_trade: fracción de capital a arriesgar por operación (0.01 = 1%).
    """

    risk_per_trade: float = 0.01
    max_position_pct: float = 1.0  # máx. fracción del capital en una posición
    min_shares: int = 1


def compute_position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    params: PositionSizingParams | None = None,
) -> tuple[int, float, float]:
    """Calcula la cantidad de acciones según la regla de riesgo fijo.

    Returns:
        (shares, notional, risk_amount)
    """
    params = params or PositionSizingParams()
    if entry_price <= 0 or capital <= 0:
        return 0, 0.0, 0.0

    risk_per_share = abs(entry_price - stop_price)
    risk_amount = capital * params.risk_per_trade

    if risk_per_share <= 0:
        return 0, 0.0, 0.0

    shares = int(risk_amount / risk_per_share)
    shares = max(shares, params.min_shares)

    notional = shares * entry_price
    max_notional = capital * params.max_position_pct
    if notional > max_notional:
        shares = int(max_notional / entry_price)
        shares = max(shares, params.min_shares)
        notional = shares * entry_price

    return shares, notional, shares * risk_per_share


def risk_reward_ratio(
    entry_price: float, stop_price: float, target_price: float
) -> float:
    """Ratio beneficio/riesgo (R/R). 0 si no es válido."""
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    if risk <= 0:
        return 0.0
    return reward / risk