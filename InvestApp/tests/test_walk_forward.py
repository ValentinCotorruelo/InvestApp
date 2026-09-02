"""Tests del modo walk-forward: calentamiento 2y / operación sin mirar el futuro."""
import numpy as np
import pandas as pd
import pytest

from investapp.backtest.engine import BacktestEngine, _offset_days
from investapp.strategies.confluence import ConfluenceStrategy
from investapp.strategies.hybrid import HybridStrategy


def make_df(n=600, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2025-12-31", periods=n)
    t = np.linspace(0, 6 * np.pi, n)
    close = 100 * np.exp(0.2 * t / n) + 10 * np.sin(t) + rng.normal(0, 0.5, n)
    close = np.maximum(close, 5.0)
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, n),
            "High": close + np.abs(rng.normal(0, 0.7, n)) + 0.2,
            "Low": close - np.abs(rng.normal(0, 0.7, n)) - 0.2,
            "Close": close,
            "Volume": rng.integers(800_000, 2_500_000, n),
        },
        index=idx,
    )
    df.attrs["ticker"] = "TEST"
    return df


def test_offset_days():
    assert _offset_days("2y") == 730
    assert _offset_days("18m") == 540
    assert _offset_days("45d") == 45
    assert _offset_days("2w") == 14
    with pytest.raises(ValueError):
        _offset_days("x")


def test_walk_forward_no_trades_in_warmup():
    df = make_df()
    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(df, ConfluenceStrategy(), warmup="2y")

    assert result.test_start is not None
    # Ningún trade antes del inicio de la ventana de operación
    if result.trades:
        first_entry = result.trades[0].entry_date
        assert first_entry >= result.test_start

    # La curva de equity arranca en el capital inicial (sin operar en calentamiento)
    assert result.equity.iloc[0] == pytest.approx(10_000.0)
    # Solo cubre la ventana de operación
    warmup_bars = int(df.index.searchsorted(result.test_start, side="left"))
    assert len(result.equity) == len(df) - warmup_bars


def test_walk_forward_decisions_registro_dia_a_dia():
    df = make_df()
    engine = BacktestEngine(initial_capital=10_000)
    for strat_cls in (ConfluenceStrategy, HybridStrategy):
        result = engine.run(df, strat_cls(), warmup="2y")
        assert len(result.decisions) == len(result.equity), strat_cls.name
        for d in result.decisions:
            assert d.action in ("entry", "exit", "hold"), strat_cls.name
            assert d.date >= result.test_start, strat_cls.name
            assert d.indicators, strat_cls.name
        cols = result.decisions_df().columns.tolist()
        assert "date" in cols and "action" in cols and "RSI" in cols


def test_walk_forward_warmup_insuficiente():
    df = make_df(n=300)  # ~14 meses de datos: no alcanza para calentar + operar
    engine = BacktestEngine()
    with pytest.raises(ValueError):
        engine.run(df, ConfluenceStrategy(), warmup="1y")


def test_walk_forward_permite_entradas_solo_en_operacion():
    """No debe haber entradas antes de la ventana de operación incluso con señales."""
    df = make_df()
    engine = BacktestEngine()
    result = engine.run(df, ConfluenceStrategy(), warmup="2y")
    if result.events:
        entry_dates = [e["date"] for e in result.events if e["type"] == "entry"]
        assert all(d >= result.test_start for d in entry_dates)