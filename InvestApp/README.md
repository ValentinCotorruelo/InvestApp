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
| `investapp.strategies` | 5 estrategias con puntos de compra/venta explícitos (4 multi-indicador + S/R pura) |
| `investapp.risk` | Tamaño de posición por riesgo (1%), stop-loss, take-profit, trailing, cooldowns, R/R |
| `investapp.backtest` | Motor de backtest diario long-only + métricas (Sharpe, Sortino, drawdown, win rate) |
| `investapp.backtest.walk_forward` | Barrido multi-activo × estrategia (2y calibración / 3y operación) + reporte HTML interactivo |
| `investapp.backtest.walk_forward_engine` | Motor público multi-activo con ejecución al open de t+1 (`next_open`), resultado por activo y prueba anti-look-ahead |
| `investapp.screener` | Barrido de muchas acciones y reportes CLI/HTML/CSV |

## Estrategias

Las estrategias determinan puntos de compra/venta con reglas explícitas:

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

5. **Soportes y Resistencias** (`support_resistance` / `sr`)
   Pura, sin otros indicadores: compra por **rebote en soporte** (cierre sobre un
   soporte tocado) o **ruptura de resistencia** (cierre con margen > 1%). Salidas
   por pérdida del soporte o aproximación a la resistencia objetivo. Solo precio
   y niveles. Reutilizada tal cual por el motor de walk-forward.

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
- `fill="open" | "close" | "next_open"` define dónde se ejecuta la orden:
  - `"next_open"` (recomendado): señal calculada con el cierre de `t`, orden
    ejecutada al **open de `t+1`**. Evita el sesgo de "comprar al precio que
    generó la señal" y es el modo por defecto del motor multi-activo.
  - `"open"` / `"close"`: ejecuta el mismo día de la señal.
- Sin lookahead garantizado por tests de *prefix-equivalence* (evaluar un día
  con datos hasta ese día == evaluarlo sin tener el futuro).

## Motor multi-activo (walk-forward, open de t+1)

Interfaz pública que corre una estrategia día a día sobre **muchos activos** y
devuelve un resultado **por activo** (para comparar qué activos funcionan mejor).
Consume la estrategia tal cual (no reimplementa su lógica) con `fill="next_open"`,
una posición a la vez, comisión 0.1% por operación y tamaño de posición por riesgo
(parametrizable):

```python
from investapp.backtest.walk_forward_engine import run_walk_forward, summary_table
from investapp.strategies import SupportResistanceStrategy

results = run_walk_forward(
    activos=["AAPL", "SPY", "NVDA", "KO", "BA"],
    fecha_inicio="2021-01-04",
    fecha_fin="2024-12-31",
    estrategia_fn=lambda: SupportResistanceStrategy(),  # instancia nueva por activo
    capital_inicial=10_000,
    warmup="2y",
)
print(summary_table(results))   # tabla por activo
```

- Cada operación queda registrada con fecha/precio de entrada y salida, motivo
  (`stop_loss` / `take_profit` / `señal_salida`) y % de resultado; una posición
  abierta al final se marca como `open` (valuada al cierre del último día).
- Activos sin datos suficientes se saltan sin romper la corrida.
- La **prueba anti-look-ahead** corta el histórico en una fecha intermedia y
  exige señales idénticas hasta esa fecha (ver `tests/test_walk_forward_multi.py`).

```bash
python examples\ejemplo_walk_forward.py   # 5 activos reales + anti-look-ahead + gráfico
```

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