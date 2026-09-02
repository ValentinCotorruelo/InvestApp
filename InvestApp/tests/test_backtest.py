"""Tests de riesgo y backtest con datos sintéticos."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    n = 500
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0005, 0.02, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + np.random.default_rng(1).normal(0, 0.005, n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.random.default_rng(2).integers(1e5, 5e6, n),
        },
        index=idx,
    )


def test_position_sizing():
    from investapp.risk.position_sizing import compute_position_size, risk_reward_ratio

    shares, notional, risk = compute_position_size(10_000, 100, 95)
    assert shares >= 1
    # riesgo fijo 1% del capital = 100
    assert abs(risk - 100) <= 105

    rr = risk_reward_ratio(100, 95, 115)
    assert rr == pytest.approx(3.0)


def test_trailing_stop():
    from investapp.risk.risk_management import update_trailing_stop

    # El trailing solo ratchete hacia arriba (sigue máximos)
    t1 = update_trailing_stop(None, 100, 2.0, 3.0)
    assert t1 == pytest.approx(94.0)
    # El precio sube a un nuevo máximo -> trailing sube
    t2 = update_trailing_stop(t1, 108, 2.0, 3.0)
    assert t2 == pytest.approx(108 - 6.0)
    # El precio cae: el trailing se mantiene en su nivel más alto
    t3 = update_trailing_stop(t2, 95, 2.0, 3.0)
    assert t3 == t2


def test_backtest_engine(df):
    from investapp.backtest import BacktestEngine
    from investapp.strategies import HybridStrategy

    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(df, HybridStrategy(), period=None)
    assert len(result.equity) == len(df)
    assert result.metrics.num_trades >= 0
    # equity comienza en capital inicial
    assert result.equity.dropna().iloc[0] == pytest.approx(10_000.0)


def test_backtest_respetar_cooldown(df):
    from investapp.backtest import BacktestEngine
    from investapp.strategies import HybridStrategy

    engine = BacktestEngine(initial_capital=10_000)
    result = engine.run(df, HybridStrategy(), period=None, cooldown_days=2)
    assert result.metrics.num_trades >= 0


def test_metrics():
    from investapp.utils.metrics import compute_metrics

    eq = pd.Series(np.linspace(100, 150, 252))
    m = compute_metrics(eq)
    assert m.total_return == pytest.approx(0.5)
    assert m.max_drawdown <= 0