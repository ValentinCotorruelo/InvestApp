# InvestApp

Analizador de inversiones con **backtesting** y **market screener** para acciones.
Dado un activo, descarga datos históricos (Yahoo Finance), evalúa estrategias
multi-indicador complejas (puntos de compra/venta) y permite barrer cientos de
acciones para devolver las que están **en momento de compra o de venta**.

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python 3.10+.

## Módulos

| Módulo | Descripción |
|--------|-------------|
| `investapp.data` | Descarga y normalización de datos (`yfinance`) + gestión de lista de tickers |
| `investapp.indicators` | RSI, MACD, ADX, ATR, EMA/SMA, Estocástico, Parabólico SAR, Bollinger, Keltner, OBV, CMF, MFI, VWAP, ROC |
| `investapp.market_structure` | Pivotes (zigzag), Soportes/Resistencias ponderados, Retrocesos y Extensiones de Fibonacci, clasificación de tendencia |
| `investapp.strategies` | 4 estrategias multi-indicador con puntos de compra/venta explícitos |
| `investapp.risk` | Tamaño de posición por riesgo (1%), stop-loss, take-profit, trailing, cooldowns, R/R |
| `investapp.backtest` | Motor de backtest diario long-only + métricas (Sharpe, Sortino, drawdown, win rate) |
| `investapp.backtest.walk_forward` | Barrido multi-activo × estrategia (2y calibración / 3y operación) + reporte HTML interactivo |
| `investapp.screener` | Barrido de muchas acciones y reportes CLI/HTML/CSV |

## Estrategias

Las 4 estrategias combinan **muchos indicadores** y determinan puntos de
compra/venta con reglas explícitas:

1. **Trend Follower** (`trend_follower`)
   Sigue tendencias fuertes: EMA200 alcista, ADX>25 con +DI>-DI, EMA20>EMA50
   (golden cross), MACD sobre señal y sobre cero, ruptura de resistencia con
   volumen. SL por soporte/ATR, TP por extensión de Fibonacci.

2. **Mean Reversion** (`mean_reversion`)
   Compra en rangos laterales (ADX<20): toque de banda inferior de Bollinger,
   RSI<40, estocástico sobrevendido con giro alcista, apoyo en soporte o en
   retracement de Fibonacci 0.618/0.786. Sale en banda media/superior o cuando
   el rango se rompe.

3. **Confluence** (`confluence`)
   Sistema de voto ponderado sobre 6 factores (tendencia estructural, RSI sano,
   MACD alcista, cruce de EMA, soporte/resistencia, zona de Fibonacci). Opera con
   score ≥ 6/9, volumen > 1.0 y R/R ≥ 1.5. Vende al perder la confluencia.

4. **Hybrid** (`hybrid`)
   Detecta el régimen de mercado y aplica la sub-estrategia adecuada:
   ADX>25 → trend follower; ADX<20 → mean reversion; 20-25 → confluence.

## Backtest

```python
from investapp.backtest import BacktestEngine
from investapp.strategies import TrendFollowerStrategy

engine = BacktestEngine(initial_capital=10_000)
result = engine.run("AAPL", TrendFollowerStrategy(), period="3y")
result.report()        # métricas completas
result.list_trades()   # detalle de operaciones
result.plot(filename="backtest.png")
```

## Walk-Forward (día a día, sin mirar el futuro)

Baja el histórico, usa los primeros años solo para "madurar" indicadores y
niveles (calentamiento) y opera el resto **pasando día a día**, evaluando en
cada barra solo con los datos conocidos hasta ese momento (pivotes confirmados,
niveles S/R causales, régimen causal). Toda decisión queda registrada:

```python
from investapp.backtest import BacktestEngine
from investapp.strategies import ConfluenceStrategy

result = BacktestEngine().run(
    "AAPL", ConfluenceStrategy(), period="5y", warmup="2y"
)
result.report()            # métricas SOLO de la ventana de operación (3y)
result.decisions_df()      # paso a paso: fecha, precio, acción, señales, modo, indicadores
```

- `warmup="2y"` → los primeros 2 años son calentamiento; compras ventas empiezan después.
- La curva de equity y las métricas cubren solo la ventana de operación.
- Sin lookahead garantizado por tests de *prefix-equivalence* (evaluar un día
  con datos hasta ese día == evaluarlo sin tener el futuro).

## Barrido multi-activo + reporte interactivo (UI)

Barrer `tickers.csv` × las 4 estrategias y abrir un reporte HTML interactivo
(Plotly **embebido**, funciona sin internet) que muestra la tabla resumen y,
uno a la vez, el candlestick con entradas/salidas, volumen, equity y el
**paso a paso día a día** de cada activo:

```python
from pathlib import Path
from investapp.backtest import WalkForwardConfig, WalkForwardRunner, build_interactive_report

runner = WalkForwardRunner(WalkForwardConfig(period="5y", warmup="2y"))
wf = runner.run(tickers_file="tickers.csv")
build_interactive_report(wf, "reporte_walk_forward.html")
wf.decisiones_csv("decisiones.csv")
```

O directo:

```bash
python examples\walk_forward_ejemplo.py
```

Genera `reporte_walk_forward.html` y `decisiones.csv` (el paso a paso de todos
los activos).

## Market Screener

Barrer las acciones de `tickers.csv` y ver las señales:

```python
from investapp.screener import MarketScreener
from investapp.strategies import HybridStrategy

screener = MarketScreener(tickers_file="tickers.csv", verbose=True)
resultado = screener.scan(HybridStrategy(), period="2y")
resultado.mostrar_cli()
resultado.exportar_html("reporte_senales.html")   # reporte visual con gráficos
resultado.exportar_csv("reporte_senales.csv")

compras = resultado.filtrar("COMPRA")
ventas = resultado.filtrar("VENTA")

# Cruzar estrategias: activos en COMPRA según hybrid Y confluence
cruce = resultado.intersectar(
    screener.scan(ConfluenceStrategy()).filtrar("COMPRA")
)
```

La señal `COMPRA` indica un punto de entrada vigente con `entrada`, `stop_loss`,
`take_profit` y `R/R`. La señal `VENTA` indica una señal de salida **nueva**
(no estados crónicos).

## Lista de tickers

Edita `tickers.csv` (una acción por línea) o usa la clase `TickerList`:

```python
from investapp.data import TickerList

tl = TickerList(["AAPL", "MSFT", "NVDA"])
tl.save_csv("tickers.csv")
```

## Tests

```bash
python -m pytest tests -q
```

## Notas

- Datos diarios, long-only, secuencial (respetuoso con límites de API).
- Las señales son señales técnicas, no asesoría financiera.