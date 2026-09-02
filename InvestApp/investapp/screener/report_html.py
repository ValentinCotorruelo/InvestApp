"""Generación de reporte HTML visual con gráficos embebidos (base64)."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Optional

import pandas as pd


def _munich(ohlc, entry, sl, tp):
    """Genera un candlestick chart con niveles, devuelve PNG en base64."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        return None

    matplotlib.use("Agg", force=False)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ohlc = ohlc.tail(120)
    n = len(ohlc)
    xpos = list(range(n))

    opens = ohlc["Open"].to_numpy()
    highs = ohlc["High"].to_numpy()
    lows = ohlc["Low"].to_numpy()
    closes = ohlc["Close"].to_numpy()

    for i in xpos:
        bullish = closes[i] >= opens[i]
        color = "#26a69a" if bullish else "#ef5350"
        body_bottom = min(opens[i], closes[i])
        body_top = max(opens[i], closes[i])
        axes[0].plot([i, i], [lows[i], highs[i]], color=color, lw=1.0)
        axes[0].bar(i, max(body_top - body_bottom, 1e-6), bottom=body_bottom, color=color, width=0.6)

    if entry:
        axes[0].axhline(entry, color="blue", linestyle="--", lw=1.2, label=f"Entrada {entry:.2f}")
    if sl:
        axes[0].axhline(sl, color="red", linestyle="--", lw=1.2, label=f"SL {sl:.2f}")
    if tp:
        axes[0].axhline(tp, color="green", linestyle="--", lw=1.2, label=f"TP {tp:.2f}")

    has_levels = any(v for v in (entry, sl, tp))
    if has_levels:
        axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Precio")
    axes[0].grid(alpha=0.3)

    axes[1].bar(xpos, ohlc["Volume"].to_numpy(), color="#78909c", width=0.6)
    axes[1].set_ylabel("Volumen")
    axes[1].grid(alpha=0.3)

    if "RSI" in ohlc.columns and ohlc["RSI"].notna().any():
        axes[2].plot(xpos, ohlc["RSI"].to_numpy(), color="purple", lw=1.2)
        axes[2].axhline(30, color="red", lw=0.7, ls=":")
        axes[2].axhline(70, color="red", lw=0.7, ls=":")
        axes[2].set_ylim(0, 100)
        axes[2].set_ylabel("RSI")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].axis("off")

    step = max(1, n // 8)
    ticks = xpos[::step]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(
        [ohlc.index[i].strftime("%d-%m-%Y") for i in ticks], rotation=45, fontsize=8
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def build_html(result, title: Optional[str] = None) -> str:
    """Construye el HTML completo del reporte."""
    title = title or f"InvestApp - Reporte de Señales ({result.strategy_name})"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    compras = result.filtrar("COMPRA").signals
    ventas = result.filtrar("VENTA").signals

    def _badge(senal: str) -> str:
        color = "#1b5e20" if senal == "COMPRA" else ("#b71c1c" if senal == "VENTA" else "#546e7a")
        return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:10px">{senal}</span>'

    def _table(rows, show_graphs: bool) -> str:
        parts = []
        for r in rows:
            reasons = "<br>".join(f"· {reason}" for reason in r.reasons) if r.reasons else "—"
            img = ""
            if show_graphs and r.df is not None:
                data_uri = _munich(r.df, r.entrada, r.stop_loss, r.take_profit)
                if data_uri:
                    img = f'<img src="{data_uri}" style="max-width:100%;margin-top:8px;"/>'
            parts.append(
                f"""
                <div style="border:1px solid #ddd;border-radius:8px;padding:14px;margin:10px 0;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                    <strong style="font-size:16px;">{r.ticker}</strong>
                    {_badge(r.senal)}
                    <span style="color:#555;">Régimen: {r.regime}</span>
                    <span style="color:#555;">Precio: <b>{r.precio:.2f}</b></span>
                    <span style="color:#555;">Fuerza: {r.strength:.2f}</span>
                    <span style="color:#555;">R/R: {r.rr:.2f}</span>
                  </div>
                  <table style="border-collapse:collapse;margin-top:8px;font-size:13px;">
                    <tr>
                      <td style="padding:2px 12px 2px 0;"><b>Entrada</b></td><td style="padding:2px 12px;">{r.entrada:.2f}</td>
                      <td style="padding:2px 12px 2px 0;"><b>Stop-loss</b></td><td style="padding:2px 12px;color:#b71c1c;">{r.stop_loss:.2f}</td>
                      <td style="padding:2px 12px 2px 0;"><b>Take-profit</b></td><td style="padding:2px 12px;color:#1b5e20;">{r.take_profit if r.take_profit else "—"}</td>
                    </tr>
                  </table>
                  <div style="margin-top:6px;color:#333;font-size:13px;">{reasons}</div>
                  {img}
                </div>
                """
            )
        if not parts:
            return '<p style="color:#888;">Sin resultados en esta categoría.</p>'
        return "".join(parts)

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>{title}</title>
      <style>
        body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#f5f5f5; margin:0; padding:20px; }}
        h1 {{ font-size:22px; margin:0 0 4px 0; }}
        .meta {{ color:#666; font-size:13px; margin-bottom:20px; }}
        .section {{ margin:24px 0; }}
        .section h2 {{ font-size:17px; border-bottom:2px solid #26a69a; padding-bottom:6px; }}
        .summary {{ display:flex; gap:20px; margin:16px 0; }}
        .card {{ background:#fff; border-radius:8px; padding:16px 24px; box-shadow:0 1px 3px rgba(0,0,0,0.1); text-align:center; }}
        .card .num {{ font-size:26px; font-weight:bold; }}
        .card .label {{ color:#666; font-size:12px; }}
        a {{ color:#00695c; }}
        .footer {{ margin-top:30px; color:#999; font-size:12px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <div class="meta">Generado el {generated}</div>

      <div class="summary">
        <div class="card"><div class="num" style="color:#1b5e20;">{len(compras)}</div><div class="label">COMPRA</div></div>
        <div class="card"><div class="num" style="color:#b71c1c;">{len(ventas)}</div><div class="label">VENTA</div></div>
        <div class="card"><div class="num">{len(result.signals)}</div><div class="label">Total señales</div></div>
      </div>

      <div class="section">
        <h2>En momento de COMPRA ({len(compras)})</h2>
        {_table(compras, show_graphs=True)}
      </div>

      <div class="section">
        <h2>En momento de VENTA ({len(ventas)})</h2>
        {_table(ventas, show_graphs=True)}
      </div>

      <div class="section">
        <h2>Sin señal ({len(result) - len(compras) - len(ventas)})</h2>
        <p style="color:#999;">Los activos sin señal no se listan por defecto. Ejecuta el barrido con verbose=True para verlos.</p>
      </div>

      <div class="footer">InvestApp · Analizador de inversiones con backtesting y market screener.</div>
    </body>
    </html>
    """
    return html