"""Tests de la estrategia de Soportes y Resistencias.

Criterio de éxito: señales con frecuencia razonable (no cero en 3 años),
niveles coherentes (SL < entry < TP), R/R >= 1.5 y causalidad (sin lookahead).
"""
import numpy as np
import pandas as pd
import pytest

from investapp.strategies.support_resistance_strategy import (
    MIN_RR,
    SupportResistanceStrategy,
)


def make_df(n=800, seed=3):
    """Serie determinística con tendencia + oscilaciones (simula precio real)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    t = np.linspace(0, 5 * np.pi, n)
    close = 100 * np.exp(0.3 * t / n) + 15 * np.sin(t) + rng.normal(0, 0.6, n)
    close = np.maximum(close, 5.0)
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, n),
            "High": close + np.abs(rng.normal(0, 0.8, n)) + 0.2,
            "Low": close - np.abs(rng.normal(0, 0.8, n)) - 0.2,
            "Close": close,
            "Volume": rng.integers(800_000, 2_500_000, n),
        },
        index=idx,
    )
    df.attrs["ticker"] = "TEST"
    return df


def _count_signals(strategy, df):
    """Cuenta días con señal de compra recorriendo toda la serie."""
    strategy.prepare(df)
    positions = []
    for i in range(60, len(df)):
        if strategy.evaluate(i).signal is not None:
            positions.append(i)
    return positions


def test_sr_genera_señales_sintetico():
    idx = _count_signals(SupportResistanceStrategy(), make_df())
    assert len(idx) >= 5, f"Esperaba >=5 señales, obtuve {len(idx)}: {idx}"


def test_sr_sl_tp_coherentes():
    """Toda señal debe tener SL < entry y (TP None o TP > entry)."""
    df = make_df()
    s = SupportResistanceStrategy()
    s.prepare(df)
    for i in range(60, len(df)):
        ev = s.evaluate(i)
        if ev.signal is not None:
            sig = ev.signal
            assert sig.entry > 0
            assert 0 < sig.stop_loss < sig.entry, (
                f"SL inválido en i={i}: SL={sig.stop_loss} entry={sig.entry}"
            )


def test_sr_rr_minimo():
    """Cada señal long debe respetar R/R >= 1.5."""
    df = make_df()
    s = SupportResistanceStrategy()
    s.prepare(df)
    checked = 0
    for i in range(60, len(df)):
        ev = s.evaluate(i)
        if ev.signal is not None and ev.signal.take_profit:
            sig = ev.signal
            risk = sig.entry - sig.stop_loss
            reward = sig.take_profit - sig.entry
            if risk > 0:
                assert reward / risk >= MIN_RR - 1e-9, (
                    f"R/R < {MIN_RR} en i={i}: risk={risk:.2f} reward={reward:.2f}"
                )
                checked += 1
    assert checked > 0, "no se generó ninguna señal con TP para verificar R/R"


def test_sr_causalidad_prefix_equivalence():
    """Evaluar un día con solo datos hasta ese día o con toda la serie da
    la misma decisión: no mira el futuro."""
    df = make_df()
    s_full = SupportResistanceStrategy()
    s_full.prepare(df)

    for i in range(300, len(df) - 40, 43):
        ev_full = s_full.evaluate(i)

        s_cut = SupportResistanceStrategy()
        s_cut.prepare(df.iloc[: i + 1])
        ev_cut = s_cut.evaluate(-1)

        assert (ev_full.signal is None) == (ev_cut.signal is None), (
            f"Señal difiere en i={i}"
        )
        if ev_full.signal is not None:
            a, b = ev_full.signal, ev_cut.signal
            assert abs(a.entry - b.entry) < 1e-9
            assert abs(a.stop_loss - b.stop_loss) < 1e-9


def test_sr_no_genera_sin_niveles():
    """En una serie plana sin movimientos, no debería disparar ruido."""
    n = 300
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [100.5] * n,
            "Low": [99.5] * n,
            "Close": [100.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )
    df.attrs["ticker"] = "FLAT"
    idx = _count_signals(SupportResistanceStrategy(), df)
    assert len(idx) < 5, f"Serie plana generó ruido indebido: {len(idx)} señales"
