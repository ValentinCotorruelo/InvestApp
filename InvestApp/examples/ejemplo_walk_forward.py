"""Validación del motor de walk-forward multi-activo con la estrategia S/R.

Corre el walk-forward sobre 5 activos reales × ~4 años, reporta tabla por activo,
realiza la PRUEBA ANTI-LOOK-AHEAD (corte de histórico) y genera un gráfico de
curva de capital con puntos de entrada/salida para un activo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investapp.backtest.walk_forward_engine import (
    run_walk_forward,
    summary_table,
)
from investapp.data.fetcher import DataFetcher
from investapp.strategies import SupportResistanceStrategy

TICKERS = ["AAPL", "SPY", "NVDA", "KO", "BA"]
FECHA_INICIO = "2021-01-04"
FECHA_FIN = "2024-12-31"
WARMUP = "2y"
FUENTE_INICIO = "2018-06-01"  # fecha_inicio - warmup, para madurar niveles
CAPITAL = 10_000.0


def main():
    fetcher = DataFetcher()

    # ---- Descargar datos para todos los activos (con calentamiento previo) ----
    data = {}
    for tk in TICKERS:
        try:
            data[tk] = fetcher.fetch(tk, start=FUENTE_INICIO, end=FECHA_FIN).df
        except Exception as exc:
            print(f"[skip descarga] {tk}: {exc}")

    print(f"Descargados {len(data)}/{len(TICKERS)} activos\n")

    # ---- Correr walk-forward (next_open) ----
    results = run_walk_forward(
        list(data.keys()),
        FECHA_INICIO,
        FECHA_FIN,
        lambda: SupportResistanceStrategy(),
        capital_inicial=CAPITAL,
        data=data,
        warmup=WARMUP,
    )

    print("=== Tabla por activo (walk-forward next_open) ===")
    print(summary_table(results).to_string(index=False))
    print()

    # ---- Prueba anti-look-ahead real (cortar histórico en fecha intermedia) ----
    print("=== Prueba anti-look-ahead ===")
    aktivo = "AAPL" if "AAPL" in data else list(results)[0]
    df_full = data[aktivo]
    cut = df_full.index[len(df_full) // 2]
    df_cut = df_full.loc[:cut]

    def senales_aisladas_hasta(df, hasta):
        s = SupportResistanceStrategy()
        s.prepare(df)
        out = []
        for i in range(len(df)):
            if df.index[i] > hasta:
                break
            out.append(s._buy_logic(i) is not None)
        return out

    full_before = senales_aisladas_hasta(df_full, cut)
    cut_signals = senales_aisladas_hasta(df_cut, cut)
    ok = full_before == cut_signals
    print(f"activo={aktivo}, corte={cut.date()} ({len(df_cut)} barras)")
    print(f"señales hasta corte (histórico completo) = {sum(full_before)}")
    print(f"señales hasta corte (histórico cortado)  = {sum(cut_signals)}")
    print("¿Idénticas? ->", "SÍ, sin look-ahead bias" if ok else "NO, HAY LOOK-AHEAD")
    assert ok, "¡Look-ahead bias detectado!"
    print()

    # ---- Gráfico de curva de capital con entradas/salidas ----
    print("Generando gráfico de curva de capital...")
    _grafico_curva(results[aktivo], data, aktivo)


def _grafico_curva(res, data, ticker):
    import matplotlib.pyplot as plt

    df = data[ticker]
    eq = res.equity

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.5]},
    )

    # Panel 1: precio + entradas/salidas
    axes[0].plot(df.index, df["Close"], color="#334155", lw=1.1, label="Close")
    buys = [(t["entry_date"], t["entry_price"]) for t in res.trades]
    sells = [(t["exit_date"], t["exit_price"]) for t in res.trades if t["status"] == "closed"]
    if buys:
        axes[0].scatter(
            [b[0] for b in buys], [b[1] for b in buys],
            marker="^", color="#15803d", s=70, zorder=5, label=f"Entrada ({len(buys)})",
        )
    if sells:
        axes[0].scatter(
            [s[0] for s in sells], [s[1] for s in sells],
            marker="v", color="#b91c1c", s=70, zorder=5, label=f"Salida ({len(sells)})",
        )
    axes[0].set_title(f"{ticker} · Walk-forward next_open (S/R)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2: curva de capital
    axes[1].plot(eq.index, eq.values, color="#2563eb", lw=1.6, label="Capital")
    axes[1].axhline(res.capital_final, color="#94a3b8", ls="--", lw=0.8)
    axes[1].set_ylabel("Capital ($)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel 3: drawdown
    dd = eq / eq.cummax() - 1
    axes[2].fill_between(dd.index, dd.values * 100, 0, color="#dc2626", alpha=0.4)
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / f"walkforward_{ticker.lower()}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Gráfico guardado en: {out}")


if __name__ == "__main__":
    main()
