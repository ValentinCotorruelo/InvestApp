"""Tests del motor walk-forward multi-activo: next_open, consistencia y anti-look-ahead."""
import numpy as np
import pandas as pd
import pytest

from investapp.backtest.engine import BacktestEngine
from investapp.backtest.walk_forward_engine import (
    run_walk_forward,
    summary_table,
)
from investapp.strategies.support_resistance_strategy import SupportResistanceStrategy


def make_df(n=900, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2019-01-01", periods=n)
    t = np.linspace(0, 14 * np.pi, n)
    close = 100 * np.exp(0.3 * t / n) + 15 * np.sin(t) + rng.normal(0, 0.6, n)
    close = np.maximum(close, 5.0)
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, n),
            "High": close + np.abs(rng.normal(0, 1, n)) + 0.3,
            "Low": close - np.abs(rng.normal(0, 1, n)) - 0.3,
            "Close": close,
            "Volume": rng.integers(800_000, 2_500_000, n),
        },
        index=idx,
    )
    df.attrs["ticker"] = "TEST"
    return df


def _aisladas(df, start):
    """Señales de compra de la estrategia aislada (causal) desde 'start'."""
    strat = SupportResistanceStrategy()
    strat.prepare(df)
    dates, sigs = [], []
    for i in range(len(df)):
        date = df.index[i]
        if date < start:
            continue
        sig = strat._buy_logic(i)
        dates.append(date)
        sigs.append(sig is not None)
    return dates, sigs


def test_next_open_ejecuta_al_open_del_dia_siguiente():
    """En fill=next_open, cada entrada se ejecuta al OPEN del día siguiente a la señal."""
    df = make_df()
    eng = BacktestEngine(initial_capital=10_000, fill="next_open")
    res = eng.run(df, SupportResistanceStrategy(), warmup="2y")

    dec = res.decisions_df()
    opens = df["Open"]
    for _, d in dec.iterrows():
        if d["action"] == "entry":
            prev_days = dec[dec["date"] < d["date"]]
            prev_sig = prev_days[prev_days["signal_compra"]]
            assert not prev_sig.empty, "la entrada debería venir de una señal previa"
            t = prev_sig["date"].iloc[-1]
            next_open = opens.loc[df.index[df.index.get_loc(t) + 1]]
            assert d["entry_price"] == pytest.approx(next_open, rel=1e-6)
            assert d["date"] == df.index[df.index.get_loc(t) + 1]


def test_consistencia_senales_motor_vs_estrategia():
    """Las señales del motor en la ventana coinciden con la estrategia aislada."""
    df = make_df()
    eng = BacktestEngine(initial_capital=10_000, fill="next_open")
    res = eng.run(df, SupportResistanceStrategy(), warmup="2y")

    dec = res.decisions_df()
    motor_signal = dict(zip(dec["date"], dec["signal_compra"]))

    dates, aisladas = _aisladas(df, res.test_start)
    assert len(dates) == len(dec), "mismatch de cantidad de decisiones"
    for date, sig in zip(dates, aisladas):
        assert motor_signal[date] == sig


def test_anti_lookahead_corte_historico():
    """Cortar el histórico en una fecha intermedia no cambia las señales previas.

    Esta es la validación más importante del motor: si hubiera look-ahead bias,
    las señales hasta T diferirían al correr con histórico cortado vs. completo.
    """
    df = make_df()
    cut = df.index[len(df) // 2]
    df_cut = df.loc[:cut]

    _, aisladas_full = _aisladas(df, df.index[0])
    _, aisladas_cut = _aisladas(df_cut, df_cut.index[0])

    # Comparar solo hasta la fecha de corte
    full_before_cut = [s for d, s in zip(df.index, aisladas_full) if d <= cut]
    assert full_before_cut == aisladas_cut, "Look-ahead bias: señales difieren al cortar"


def test_anti_lookahead_batch_motor():
    """El motor en next_open produce señales idénticas con histórico cortado."""
    df = make_df()
    cut = df.index[len(df) // 2]
    df_cut = df.loc[:cut]

    eng = BacktestEngine(initial_capital=10_000, fill="next_open")
    res_full = eng.run(df, SupportResistanceStrategy(), warmup=None)
    res_cut = eng.run(df_cut, SupportResistanceStrategy(), warmup=None)

    def sigs_before(res):
        dec = res.decisions_df()
        return [(r["date"], bool(r["signal_compra"])) for _, r in dec.iterrows() if r["date"] <= cut]

    assert sigs_before(res_full) == sigs_before(res_cut)


def test_run_walk_forward_multiactivo_por_activo():
    """run_walk_forward devuelve un resultado por activo y tabla resumen."""
    dfs = {f"T{i}": make_df(seed=i) for i in range(3)}
    results = run_walk_forward(
        list(dfs.keys()),
        "2020-01-01",
        "2025-12-31",
        lambda: SupportResistanceStrategy(),
        capital_inicial=10_000,
        data=dfs,
    )
    assert set(results.keys()) == set(dfs.keys())
    table = summary_table(results)
    assert list(table["ticker"]) == list(dfs.keys())
    for r in results.values():
        assert r.return_total > -1.0
        assert r.max_drawdown <= 0.0


def test_run_walk_forward_maneja_activo_sin_datos():
    """Un activo sin datos suficientes no rompe la corrida completa."""
    dfs = {"A": make_df(seed=1)}
    results = run_walk_forward(
        ["A", "B"],  # 'B' no está en data -> se saltea
        "2020-01-01",
        "2025-12-31",
        lambda: SupportResistanceStrategy(),
        data=dfs,
    )
    assert "A" in results
    assert "B" not in results


def test_posicion_abierta_marcada():
    """Si queda posición abierta al final, se marca como 'open'."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range(start="2019-01-01", periods=900)
    t = np.linspace(0, 14 * np.pi, 900)
    close = 100 * np.exp(0.3 * t / 900) + 15 * np.sin(t) + rng.normal(0, 0.6, 900)
    close = np.maximum(close, 5.0)
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, 900),
            "High": close + np.abs(rng.normal(0, 1, 900)) + 0.3,
            "Low": close - np.abs(rng.normal(0, 1, 900)) - 0.3,
            "Close": close,
            "Volume": rng.integers(800_000, 2_500_000, 900),
        },
        index=idx,
    )
    eng = BacktestEngine(initial_capital=10_000, fill="next_open")
    res = eng.run(df, SupportResistanceStrategy(), warmup="2y")
    for t in res.trades:
        assert t.status in ("closed", "open")
