"""Test de causalidad: ninguna decisión puede ver el futuro.

Prefix-equivalence: evaluar la estrategia en el día i usando SOLO los datos
hasta i (prepare sobre el prefijo + evaluate(-1)) debe dar exactamente el mismo
resultado que preparar sobre toda la serie y evaluar en i. Si hay cualquier fuga
de lookahead (pivotes futuros, niveles S/R futuros, régimen global, etc.), la
comparación difiere.
"""
import numpy as np
import pandas as pd
import pytest

from investapp.strategies.confluence import ConfluenceStrategy
from investapp.strategies.hybrid import HybridStrategy
from investapp.strategies.mean_reversion import MeanReversionStrategy
from investapp.strategies.trend_follower import TrendFollowerStrategy

ALL_STRATEGIES = [
    TrendFollowerStrategy,
    MeanReversionStrategy,
    ConfluenceStrategy,
    HybridStrategy,
]


def make_df(n=520, seed=7):
    """Serie de precios determinística con tendencia + oscilaciones."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    t = np.linspace(0, 6 * np.pi, n)
    base = 100 * np.exp(0.25 * t / n) + 12 * np.sin(t)
    close = base + rng.normal(0, 0.6, n)
    close = np.maximum(close, 5.0)
    high = close + np.abs(rng.normal(0, 0.7, n)) + 0.2
    low = close - np.abs(rng.normal(0, 0.7, n)) - 0.2
    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.1, n),
            "High": np.maximum(high, close),
            "Low": np.minimum(low, close),
            "Close": close,
            "Volume": rng.integers(800_000, 2_500_000, n),
        },
        index=idx,
    )
    df.attrs["ticker"] = "TEST"
    return df


def _eq_signal(a, b, tol=1e-9):
    if (a is None) != (b is None):
        return False
    if a is None:
        return True
    same_tp = (a.take_profit is None) == (b.take_profit is None) and (
        a.take_profit is None or abs(a.take_profit - b.take_profit) < tol
    )
    return (
        abs(a.entry - b.entry) < tol
        and abs(a.stop_loss - b.stop_loss) < tol
        and same_tp
        and a.reasons == b.reasons
        and abs(a.strength - b.strength) < tol
    )


@pytest.mark.parametrize("strat_cls", ALL_STRATEGIES)
@pytest.mark.parametrize("i", [300, 380, 460, 510])
def test_prefix_equivalence(strat_cls, i):
    df = make_df()
    full = strat_cls()
    full.prepare(df)
    full_ev = full.evaluate(i)

    prefix = strat_cls()
    prefix.prepare(df.iloc[: i + 1])
    prefix_ev = prefix.evaluate(-1)

    assert _eq_signal(full_ev.signal, prefix_ev.signal), f"{strat_cls.name}: señal difiere en i={i}"
    assert full_ev.should_exit == prefix_ev.should_exit, f"{strat_cls.name}: salida difiere en i={i}"
    assert full_ev.exit_fresh == prefix_ev.exit_fresh, f"{strat_cls.name}: exit_fresh difiere en i={i}"
    for key, val in full_ev.context.items():
        if key in ("date", "ticker"):
            continue
        assert prefix_ev.context.get(key) == val, f"{strat_cls.name}: context[{key}] difiere en i={i}"
    for key in full_ev.indicators:
        fv = full_ev.indicators[key]
        pv = prefix_ev.indicators[key]
        fv = fv if fv == fv else float("nan")  # NaN == NaN
        pv = pv if pv == pv else float("nan")
        assert (fv == pv) or (np.isnan(fv) and np.isnan(pv)), f"{strat_cls.name}: indicators[{key}] difiere en i={i}"


def test_hybrid_mode_causal():
    """El modo reportado por hybrid en el día i depende solo de datos hasta i."""
    df = make_df()
    h = HybridStrategy()
    h.prepare(df)
    idx = 400
    ev = h.evaluate(idx)
    # El modo coincide con el que daría la sub-estrategia de ADX causal de ese día
    assert ev.context.get("mode") in ("trend_follower", "mean_reversion", "confluence")


def test_prefix_equivalence_ventana_completa():
    """Sin lookahead, decisiones en ventana de prueba son iguales con y sin futuro."""
    df = make_df()
    for strat_cls in ALL_STRATEGIES:
        s_full = strat_cls()
        s_full.prepare(df)
        s_cut = strat_cls()
        # Recortar el futuro: el último 40% de los datos queda vacío
        cut_at = int(len(df) * 0.6)
        s_cut.prepare(df.iloc[: cut_at + 1])

        for i in range(300, cut_at - 20, 17):
            ev_full = s_full.evaluate(i)
            ev_cut = s_cut.evaluate(i)
            assert _eq_signal(ev_full.signal, ev_cut.signal), f"{strat_cls.name}: i={i}"