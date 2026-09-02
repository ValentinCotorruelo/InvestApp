"""Ejemplo de validación de la estrategia de Soportes y Resistencias.

Descarga datos reales de varios activos (mixtos: tech, índice, defensiva),
corre la estrategia y reporta el conteo de señales para verificar que genera
señales con frecuencia razonable (no cero en 3 años, ni ruido diario).
Por último genera un gráfico con los niveles y las señales de un activo.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investapp.data.fetcher import DataFetcher
from investapp.strategies import SupportResistanceStrategy

TICKERS = ["AAPL", "SPY", "NVDA", "KO", "BA"]  # tech, índice, semivolátil, defensiva, cíclica
PERIOD = "4y"


def contar_senales(df, estrategia):
    """Simula un portafolio LONG: cuenta entradas reales y salidas reales.

    - Entrada: hay señal de compra y no hay posición.
    - Salida: hay posición y la estrategia marca should_exit (o se llega a SL).
    Así el conteo refleja operaciones completadas, no ruido de evaluación diaria.
    """
    estrategia.prepare(df)
    compras = []
    ventas = []
    en_posicion = False
    entry = 0.0
    for i in range(len(df)):
        ev = estrategia.evaluate(i)
        close = float(df["Close"].iloc[i])

        # Salir primero (si hay posición y hay señal/stop)
        if en_posicion:
            stop = ev.signal.stop_loss if False else None
            salir = ev.should_exit
            if salir or close < entry * 0.90:
                ventas.append((df.index[i], ev.context.get("exit_reasons", []) or ["stop"]))
                en_posicion = False

        # Entrar (solo si no hay posición)
        if not en_posicion and ev.signal is not None:
            compras.append((df.index[i], ev.signal.entry, ev.signal.reasons[:1]))
            entry = ev.signal.entry
            en_posicion = True

    return compras, ventas


def main():
    fetcher = DataFetcher()
    rows = []
    for ticker in TICKERS:
        try:
            md = fetcher.fetch(ticker, period=PERIOD)
        except Exception as exc:
            print(f"[error] {ticker}: {exc}")
            continue
        df = md.df
        compras, ventas = contar_senales(df, SupportResistanceStrategy())
        years = df.index[-1].year - df.index[0].year + 1
        rows.append(
            {
                "ticker": ticker,
                "barras": len(df),
                "años": years,
                "compras": len(compras),
                "ventas": len(ventas),
                "frec_compra/día": round(len(compras) / max(len(df), 1), 4),
            }
        )
        print(f"{ticker}: {len(compras)} compras, {len(ventas)} ventas "
              f"(~{len(compras)/max(len(df),1):.4f}/día) sobre {len(df)} barras")

    print("\n=== Resumen validación ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # Gráfico visual del último activo con niveles + señales
    if rows:
        print("\nGenerando gráfico de validación visual...")
        _graficar(fetcher, "AAPL")


def _graficar(fetcher, ticker):
    import matplotlib.pyplot as plt

    md = fetcher.fetch(ticker, period=PERIOD)
    df = md.df
    s = SupportResistanceStrategy()
    s.prepare(df)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df["Close"], color="#334155", lw=1.2, label="Close")

    # Niveles S/R vigentes (último día) como líneas horizontales
    for lvl in s._sr_levels_at(df.index[-1]):
        color = "#16a34a" if lvl.kind == "support" else "#dc2626"
        ax.axhline(lvl.price, color=color, alpha=0.5, lw=1, ls="--",
                   label=f"{lvl.kind} {lvl.price:.1f} ({lvl.touches} toques)")

    # Marcadores de señales a lo largo de la serie
    bxs, buys, sxs, sells = [], [], [], []
    for i in range(len(df)):
        ev = s.evaluate(i)
        if ev.signal is not None:
            bxs.append(df.index[i])
            buys.append(ev.signal.entry)
        if ev.should_exit:
            sxs.append(df.index[i])
            sells.append(df["Close"].iloc[i])
    ax.scatter(bxs, buys, marker="^", color="#15803d", s=90, zorder=5,
               label=f"Compra ({len(bxs)})")
    ax.scatter(sxs, sells, marker="v", color="#b91c1c", s=90, zorder=5,
               label=f"Venta ({len(sxs)})")

    ax.set_title(f"{ticker} · Soportes y Resistencias ({PERIOD})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(__file__).resolve().parents[1] / f"grafico_sr_{ticker.lower()}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Gráfico guardado en: {out}")


if __name__ == "__main__":
    main()
