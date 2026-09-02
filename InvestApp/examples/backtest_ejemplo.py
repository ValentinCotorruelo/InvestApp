"""Ejemplo de backtest: correr una estrategia sobre un activo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from investapp.backtest import BacktestEngine
from investapp.strategies import (
    ConfluenceStrategy,
    HybridStrategy,
    MeanReversionStrategy,
    TrendFollowerStrategy,
)

# 1. Backtest simple de Trend Follower sobre AAPL
engine = BacktestEngine(initial_capital=10_000)
result = engine.run("AAPL", TrendFollowerStrategy(), period="3y")
result.report()

# 2. Cruzar varias estrategias sobre el mismo activo
for strategy in [MeanReversionStrategy(), ConfluenceStrategy(), HybridStrategy()]:
    r = engine.run("AAPL", strategy, period="3y")
    print(
        f"\n{strategy.name}: trades={r.metrics.num_trades} "
        f"ret={r.metrics.total_return:.2%} sharpe={r.metrics.sharpe_ratio:.2f}"
    )

# 3. Ver el detalle de las operaciones
print("\nÚltimos trades (AAPL / trend_follower):")
print(result.list_trades(5).to_string(index=False))

# 4. Gráfico (descomentar para guardar el PNG)
result.plot(filename="backtest_aapl.png")