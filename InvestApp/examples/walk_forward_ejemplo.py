"""Walk-forward día a día: para cada activo y estrategia se baja el histórico,
se calientan los indicadores con los primeros años y se opera el resto
pasando día a día sin mirar el futuro.

Genera:
  reporte_walk_forward.html  → reporte interactivo (un activo a la vez + resumen)
  decisiones.csv            → el paso a paso de TODOS los activos/estrategias
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investapp.backtest import (
    WalkForwardConfig,
    WalkForwardRunner,
    build_interactive_report,
)

ROOT = Path(__file__).resolve().parents[1]

# 2 años de calentamiento + 3 años de operación (sobre 5 años descargados)
config = WalkForwardConfig(
    period="5y",
    warmup="2y",
    initial_capital=10_000,
)

runner = WalkForwardRunner(config=config)

# Opcional: limitar la lista de activos para una corrida rápida
# tickers = ["AAPL", "MSFT", "NVDA", "META", "KO"]
tickers = None  # usa tickers.csv completo

print("Corriendo walk-forward (día a día, sin mirar el futuro)...")
resultado = runner.run(
    tickers=tickers,
    tickers_file=str(ROOT / "tickers.csv"),
    tqdm=True,
)

print("\n=== Resumen comparativo ===")
print(resultado.summary_df().to_string(index=False))

html = build_interactive_report(resultado, str(ROOT / "reporte_walk_forward.html"))
resultado.decisiones_csv(str(ROOT / "decisiones.csv"))
print(f"\nReporte interactivo: {html}")
print(f"Paso a paso (CSV):    {ROOT / 'decisiones.csv'}")