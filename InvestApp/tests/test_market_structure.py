"""Tests de estructura de mercado: pivotes, soportes/resistencias y Fibonacci."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def swing_df():
    """Serie con dos oscilaciones claras (picos y valles visibles)."""
    n = 250
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    t = np.linspace(0, 4 * np.pi, n)
    close = 100 + 20 * np.sin(t)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_find_pivots(swing_df):
    from investapp.market_structure.pivots import find_pivots

    pivots = find_pivots(swing_df, left=5, right=5)
    kinds = [p.kind for p in pivots]
    assert "high" in kinds and "low" in kinds
    assert len(pivots) > 3


def test_support_resistance(swing_df):
    from investapp.market_structure.pivots import find_pivots
    from investapp.market_structure.support_resistance import detect_support_resistance

    pivots = find_pivots(swing_df, left=5, right=5)
    atr = pd.Series([2.0] * len(swing_df), index=swing_df.index)
    levels = detect_support_resistance(swing_df, atr, min_touches=2)
    assert len(levels) > 0
    assert all(l.kind in ("support", "resistance") for l in levels)


def test_fibonacci_retracement(swing_df):
    from investapp.market_structure.fibonacci import (
        RETRACEMENT_LEVELS,
        compute_fibonacci,
    )
    from investapp.market_structure.pivots import find_pivots, find_swings

    swings = find_swings(find_pivots(swing_df, left=5, right=5))
    up_swings = [s for s in swings if s.direction == "up"]
    assert up_swings
    fib = compute_fibonacci(up_swings[-1])

    r50 = fib.retracement_price(0.5)
    assert r50 is not None
    lo = min(fib.swing.start_price, fib.swing.end_price)
    hi = max(fib.swing.start_price, fib.swing.end_price)
    assert lo <= r50 <= hi

    # Extensión del swing up debe quedar por encima del máximo
    ext = fib.extension_price(1.272)
    assert ext is not None
    assert ext > hi * 0.99


def test_fibonacci_evento_plano():
    from investapp.market_structure.fibonacci import compute_fibonacci
    from investapp.market_structure.pivots import Swing
    from datetime import datetime

    sw = Swing(
        start_index=pd.Timestamp("2024-01-01"),
        start_price=100,
        end_index=pd.Timestamp("2024-02-01"),
        end_price=100,
        direction="up",
    )
    fib = compute_fibonacci(sw)
    assert fib.retracements == []
    assert fib.extensions == []


def test_detect_trend():
    from investapp.market_structure.trend import detect_trend, TrendContext

    n = 300
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    close = np.linspace(100, 200, n)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    from investapp.market_structure.pivots import find_pivots

    pivots = find_pivots(df, left=5, right=5)
    highs = [p for p in pivots if p.kind == "high"]
    lows = [p for p in pivots if p.kind == "low"]
    trend = detect_trend(df, highs, lows)
    assert isinstance(trend, TrendContext)
    assert trend.bullish is True or trend.regime == "range"