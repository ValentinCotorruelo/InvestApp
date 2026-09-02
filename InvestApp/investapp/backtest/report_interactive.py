"""Reporte HTML interactivo del walk-forward.

Genera un único archivo autocontenido (plotly.js embebido, funciona offline)
con:
  1. Tabla resumen de todos los activos x estrategias.
  2. Selector que muestra UN activo x estrategia a la vez: candlestick con
     entradas/salidas, volumen, curva de equity y la tabla paso a paso
     (día a día) con la decisión de cada día.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

import pandas as pd

from .walk_forward import WalkForwardResult


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{nd}f}"


def build_interactive_report(
    wf: WalkForwardResult,
    path: str,
    tickers: Optional[Iterable[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> str:
    """Construye el reporte HTML interactivo y devuelve la ruta generada."""
    try:
        import plotly.graph_objects as go  # noqa: F401  (validar instalación)
    except ImportError as exc:
        raise RuntimeError(
            "plotly no está instalado. Ejecutá: pip install plotly"
        ) from exc

    if tickers is None:
        tickers = wf.tickers
    if strategies is None:
        strategies = wf.strategies
    tickers = list(tickers)
    strategies = list(strategies)

    selected = [
        r for r in wf.results
        if r.ticker in set(tickers) and r.strategy_name in set(strategies)
    ]

    # ---- 1. Tabla resumen ----
    summary_rows = []
    for r in selected:
        m = r.metrics
        signal_actual = ""
        last = r.decisions[-1] if r.decisions else None
        if last:
            if last.signal_compra:
                signal_actual = "COMPRA"
            elif last.signal_venta:
                signal_actual = "VENTA"
        badge = (
            f'<span class="badge b-{signal_actual.lower() if signal_actual else "none"}">'
            f'{signal_actual or "ninguna"}</span>'
        )
        summary_rows.append(
            "<tr>"
            f"<td>{r.ticker}</td>"
            f"<td>{r.strategy_name}</td>"
            f"<td>{_fmt(m.total_return * 100)}%</td>"
            f"<td>{_fmt(m.sharpe_ratio)}</td>"
            f"<td>{_fmt(m.max_drawdown * 100)}%</td>"
            f"<td>{_fmt(m.win_rate * 100)}%</td>"
            f"<td>{len(r.trades)}</td>"
            f"<td>{len(r.decisions)}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )
    summary_table = (
        "<table><thead><tr>"
        "<th>Activo</th><th>Estrategia</th><th>Retorno</th><th>Sharpe</th>"
        "<th>Max DD</th><th>Win rate</th><th>Trades</th><th>Días</th><th>Señal actual</th>"
        "</tr></thead><tbody>"
        + "".join(summary_rows)
        + "</tbody></table>"
    )

    # ---- 2. Datos por activo (precios una sola vez por ticker) ----
    prices_json: dict[str, str] = {}
    overlays_json: dict[str, dict[str, str]] = {}
    tables_json: dict[str, str] = {}
    first_key: Optional[str] = None

    for ticker, grp in _group_by_ticker(selected):
        _prices_blob(ticker, grp, prices_json)
        for r in grp:
            key = f"{ticker}||{r.strategy_name}"
            overlays_json.setdefault(ticker, {})[r.strategy_name] = _overlay_blob(r)
            tables_json[key] = _decisions_table(r)
            if first_key is None:
                first_key = key

    # ---- 3. JS que construye cada gráfico al seleccionarlo ----
    tk, st = (first_key or "||").split("||")
    js = """
const PRICES = %(PRICES)s;
const OVERLAYS = %(OVERLAYS)s;
const TABLES = %(TABLES)s;

function render(el, ticker, strategy){
  const p = PRICES[ticker];
  if (!p || !p.dates) { el.innerHTML = '<p>Sin datos.</p>'; return; }
  const up = p.close.map((c,i)=>c>=p.open[i]?1:0);
  const data = [
    { type:'candlestick', name:ticker, x:p.dates, open:p.open, high:p.high,
      low:p.low, close:p.close, yaxis:'y',
      increasing:{line:{color:'#16a34a'}}, decreasing:{line:{color:'#dc2626'}} },
    { type:'bar', name:'Volumen', x:p.dates, y:p.volume, yaxis:'y2',
      marker:{color: up.map(u=>u?'rgba(22,163,74,0.40)':'rgba(220,38,38,0.40)')} },
  ];
  const ov = (OVERLAYS[ticker]||{})[strategy];
  if (ov){
    if (ov.entries && ov.entries.length) data.push(
      { type:'scatter', mode:'markers', name:'Entrada', x:ov.entries.map(e=>e.d),
        y:ov.entries.map(e=>e.p), yaxis:'y',
        marker:{symbol:'triangle-up', size:13, color:'#15803d', line:{color:'#000', width:1}},
        text:ov.entries.map(e=>e.r||'compra'), hoverinfo:'x+y+text' });
    if (ov.exits && ov.exits.length) data.push(
      { type:'scatter', mode:'markers', name:'Salida', x:ov.exits.map(e=>e.d),
        y:ov.exits.map(e=>e.p), yaxis:'y',
        marker:{symbol:'x', size:13, color:'#b91c1c', line:{color:'#000', width:1}},
        text:ov.exits.map(e=>e.r||'venta'), hoverinfo:'x+y+text' });
    if (ov.equity && ov.equity.length) data.push(
      { type:'scatter', mode:'lines', name:'Equity', x:ov.equity.map(e=>e.d),
        y:ov.equity.map(e=>e.v), yaxis:'y3',
        line:{color:'#2563eb', width:2} });
  }
  const layout = {
    height: 640, title: ticker + ' · ' + strategy, dragmode:'zoom',
    grid:{rows:3, columns:1, pattern:'independent'},
    xaxis:{rangeslider:{visible:false}, domain:[0,1]},
    yaxis:{domain:[0.55,1]}, yaxis2:{domain:[0.30,0.50], title:'Volumen'},
    yaxis3:{domain:[0.0,0.22], title:'Equity'},
    showlegend:true, margin:{t:60,l:60,r:20,b:40}, hovermode:'x unified' };
  if (el.__plotly__) Plotly.react(el, data, layout);
  else { Plotly.newPlot(el, data, layout); el.__plotly__ = true; }
  const table = TABLES[ticker+'||'+strategy];
  document.getElementById('divDecisiones').innerHTML = table || '<p>Sin decisiones.</p>';
}

function selectKeys(){
  const t = document.getElementById('selTicker').value;
  const s = document.getElementById('selStrategy').value;
  render(document.getElementById('divPlot'), t, s);
}

window.addEventListener('load', function(){
  const t = document.getElementById('selTicker').value;
  const s = document.getElementById('selStrategy').value;
  render(document.getElementById('divPlot'), t, s);
});
""" % {
        "PRICES": json.dumps(prices_json),
        "OVERLAYS": json.dumps(overlays_json),
        "TABLES": json.dumps(tables_json),
    }

    # ---- 4. Armado final del HTML ----
    selects = (
        "<div class='controles'>"
        "<label>Activo: <select id='selTicker' onchange='selectKeys()'>"
        + "".join(f"<option>{t}</option>" for t in tickers)
        + "</select></label>"
        "<label>Estrategia: <select id='selStrategy' onchange='selectKeys()'>"
        + "".join(f"<option>{s}</option>" for s in strategies)
        + "</select></label>"
        "</div>"
    )

    plotly_tag = _plotly_lib_script_tag()

    page = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Walk-Forward InvestApp</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; color:#111; background:#f6f7f9; }}
  h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; color:#1f2937; }}
  .controles {{ margin: 14px 0; }} label {{ margin-right: 18px; }}
  select {{ padding: 6px 10px; font-size: 14px; border-radius: 6px; border: 1px solid #cbd5e1; }}
  table {{ border-collapse: collapse; width: 100%; background:#fff; font-size: 13px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: right; }}
  th {{ background:#eef2f7; }} tbody tr:hover {{ background:#f8fafc; }}
  td:nth-child(1), td:nth-child(2), th:nth-child(1), th:nth-child(2) {{ text-align: left; }}
  .badge {{ padding: 2px 8px; border-radius: 999px; font-size: 11px; }}
  .b-compra {{ background:#dcfce7; color:#15803d; }}
  .b-venta {{ background:#fee2e2; color:#b91c1c; }}
  .b-none {{ background:#e2e8f0; color:#475569; }}
  .wrap-decisiones {{ max-height: 420px; overflow: auto; border-radius: 8px; }}
  #divDecisiones table {{ font-size: 12px; }}
</style>
</head>
<body>
<h1>InvestApp · Walk-Forward (sin mirar el futuro)</h1>
<p>Calentamiento: <b>{wf.config.warmup}</b> · Total descargado: <b>{wf.config.period}</b> · Capital inicial: <b>${wf.config.initial_capital:,.0f}</b></p>
<h2>Resumen multi-activo</h2>
{summary_table}
<h2>Detalle (un activo a la vez)</h2>
{selects}
<div id="divPlot"></div>
<h2>Paso a paso (día a día)</h2>
<div class="wrap-decisiones" id="divDecisiones"><p>Seleccioná un activo.</p></div>
{plotly_tag}
<script>
{js}
</script>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return path


def _group_by_ticker(results):
    groups = {}
    for r in results:
        groups.setdefault(r.ticker, []).append(r)
    return groups.items()


def _prices_blob(ticker, group, prices_json: dict) -> None:
    df = group[0].df
    if df is None or df.empty:
        prices_json[ticker] = "{}"
        return
    prices_json[ticker] = json.dumps(
        {
            "dates": [str(d.date()) for d in df.index],
            "open": [float(x) for x in df["Open"].tolist()],
            "high": [float(x) for x in df["High"].tolist()],
            "low": [float(x) for x in df["Low"].tolist()],
            "close": [float(x) for x in df["Close"].tolist()],
            "volume": [float(x) for x in df["Volume"].fillna(0).tolist()],
        }
    )


def _overlay_blob(r) -> str:
    entries = [
        {"d": str(e["date"].date()), "p": float(e["price"]), "r": "; ".join(e.get("reasons") or [])}
        for e in r.events
        if e["type"] == "entry"
    ]
    exits = [
        {"d": str(e["date"].date()), "p": float(e["price"]), "r": e.get("reason", "")}
        for e in r.events
        if e["type"] == "exit"
    ]
    equity = [
        {"d": str(d.date()), "v": float(v)}
        for d, v in zip(r.equity.index, r.equity.tolist())
    ]
    return json.dumps({"entries": entries, "exits": exits, "equity": equity})


def _decisions_table(r) -> str:
    df = r.decisions_df()
    if df.empty:
        return "<p>Sin decisiones.</p>"
    cols = [
        "date", "action", "price", "signal_compra", "signal_venta", "regime",
        "mode", "entry_price", "stop_loss", "take_profit", "reason", "RSI", "ADX",
    ]
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells.append("<td></td>")
            elif c == "signal_compra" and v:
                cells.append("<td class='b-compra'>Sí</td>")
            elif c == "signal_venta" and v:
                cells.append("<td class='b-venta'>Sí</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:.2f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _plotly_lib_script_tag() -> str:
    """Devuelve el <script> con plotly.js embebido (offline)."""
    try:
        from plotly.offline import get_plotlyjs
    except Exception:
        return "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>"
    try:
        js = get_plotlyjs()
    except Exception:
        return "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>"
    return f"<script>\n{js}\n</script>"