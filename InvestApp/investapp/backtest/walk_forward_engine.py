"""Motor de walk-forward multi-activo (interfaz pública).

Simula, para cada activo de una lista, cómo se habría comportado una estrategia
operando día por día: señal calculada con el cierre de `t`, orden ejecutada al
**open de t+1** (sin look-ahead), una sola posición por activo, y métricas por
operación y por activo al final.

Consume la estrategia tal cual (clase `Strategy`), sin reimplementar su lógica.
Cumple la validación anti-look-ahead: cortar el histórico en una fecha intermedia
debe producir señales idénticas hasta esa fecha.

Estructura (capas):
  (a) `_simulate_asset`      -> loop de simulación día a día (abstraído en BacktestEngine).
  (b) `_manage_trades`       -> gestión de posición/operaciones (abstraído en BacktestEngine).
  (c) `_compute_metrics`     -> cálculo de métricas finales por activo.
  `run_walk_forward`         -> orquestación multi-activo (resultado por activo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..strategies.base import Strategy
from .engine import BacktestEngine, BacktestResult


@dataclass
class AssetWalkForwardResult:
    """Resultado de walk-forward de un activo."""

    ticker: str
    strategy: str
    capital_final: float
    return_total: float
    num_trades: int
    num_closed: int
    num_open: int
    win_rate: float
    avg_trade_return: float
    max_drawdown: float
    trades: list = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    decisions: list = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "estrategia": self.strategy,
            "retorno_total_pct": round(self.return_total * 100, 2),
            "trades": self.num_trades,
            "cerradas": self.num_closed,
            "abiertas": self.num_open,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "ganancia_prom_pct": round(self.avg_trade_return * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
        }


def _compute_metrics(result: BacktestResult, initial_capital: float) -> AssetWalkForwardResult:
    """Capa (c): métricas finales por activo a partir de trades + equity."""
    trades = result.trades
    closed = [t for t in trades if t.status == "closed"]
    open_trades = [t for t in trades if t.status == "open"]

    returns = [t.return_pct for t in closed] if closed else []
    wins = [r for r in returns if r > 0] if returns else []
    win_rate = float(len(wins) / len(closed)) if closed else 0.0
    avg_trade_return = float(np.mean(returns)) if returns else 0.0

    equity = result.equity
    max_dd = 0.0
    if equity is not None and len(equity) > 0:
        running_max = equity.cummax()
        dd = (equity / running_max - 1)
        max_dd = float(dd.min())

    capital_final = float(equity.iloc[-1]) if equity is not None and len(equity) > 0 else initial_capital
    return_total = capital_final / initial_capital - 1 if initial_capital > 0 else 0.0

    return AssetWalkForwardResult(
        ticker=result.ticker,
        strategy=result.strategy_name,
        capital_final=capital_final,
        return_total=return_total,
        num_trades=len(trades),
        num_closed=len(closed),
        num_open=len(open_trades),
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
        max_drawdown=max_dd,
        trades=[t.to_dict() for t in trades],
        equity=equity,
        decisions=result.decisions_df(),
    )


def _simulate_asset(
    ticker: str,
    df: pd.DataFrame,
    strategy_fn: Callable[[], Strategy],
    initial_capital: float,
    fee_rate: float,
    sizing: object,
    fill: str,
    warmup: str,
    start: Optional[str] = None,
) -> AssetWalkForwardResult:
    """Capa (a)+(b): corre la simulación día a día sobre un activo.

    `strategy_fn` devuelve una instancia FRESCA de la estrategia (para que la
    corrida sea reproducible y no comparta estado entre activos o cortes).
    `df` debe incluir el período de calentamiento previo a `start`.
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        sizing=sizing,
        fee_rate=fee_rate,
        fill=fill,
    )
    data = df.copy()
    data.attrs["ticker"] = ticker
    result = engine.run(
        data,
        strategy_fn(),
        start=start,
        warmup=warmup,
    )
    return _compute_metrics(result, initial_capital)


def run_walk_forward(
    activos: list[str],
    fecha_inicio: str,
    fecha_fin: str,
    estrategia_fn: Callable[[], Strategy],
    capital_inicial: float = 10_000.0,
    data: Optional[dict[str, pd.DataFrame]] = None,
    fee_rate: float = 0.001,
    sizing: Optional[object] = None,
    fill: str = "next_open",
    warmup: str = "2y",
) -> dict[str, AssetWalkForwardResult]:
    """Corre walk-forward sobre una lista de activos y devuelve un resultado POR ACTIVO.

    Args:
        activos: tickers a simular.
        fecha_inicio/fecha_fin: rango de la ventana OPERATIVA.
        estrategia_fn: fábrica que devuelve una instancia nueva de `Strategy`
            (se llama una vez por activo para que no comparta estado).
        capital_inicial: capital de arranque.
        data: mapeo opcional {ticker: df OHLCV} que DEBE incluir el período de
            calentamiento (`warmup`) previo a `fecha_inicio`. Si no se provee,
            se descargan vía DataFetcher con ese calentamiento.
        fee_rate: comisión fraccional por operación (ida y vuelta).
        sizing: parámetros de tamaño de posición (default: risk-per-trade 1%).
        fill: dónde ejecutar la orden. Default `"next_open"` (open de t+1, sin
            look-ahead).
        warmup: período de calentamiento previo a `fecha_inicio` (no se opera).

    Returns:
        dict {ticker: AssetWalkForwardResult}. Activos sin datos suficientes
        se omiten (no rompen la corrida).
    """
    from ..data.fetcher import DataFetcher
    from .engine import _offset_days

    fetcher = DataFetcher()
    fetch_start = (
        pd.Timestamp(fecha_inicio) - pd.Timedelta(days=_offset_days(warmup))
    ).date().isoformat()
    results: dict[str, AssetWalkForwardResult] = {}

    for ticker in activos:
        try:
            if data is not None:
                if ticker not in data:
                    print(f"  [skip] {ticker}: sin datos provistos en 'data'")
                    continue
                df = data[ticker]
            else:
                df = fetcher.fetch(ticker, start=fetch_start, end=fecha_fin).df
        except Exception as exc:
            print(f"  [skip] {ticker}: no se pudieron obtener datos ({exc})")
            continue

        # Recortar SOLO el final; el inicio conserva el calentamiento.
        df = df.loc[:fecha_fin]
        if df is None or len(df) < 100:
            print(f"  [skip] {ticker}: datos insuficientes ({0 if df is None else len(df)} filas)")
            continue

        try:
            res = _simulate_asset(
                ticker,
                df,
                estrategia_fn,
                capital_inicial,
                fee_rate,
                sizing,
                fill,
                warmup,
                start=fecha_inicio,
            )
        except Exception as exc:
            print(f"  [skip] {ticker}: fallo en simulación ({exc})")
            continue

        results[ticker] = res

    return results


def summary_table(results: dict[str, AssetWalkForwardResult]) -> pd.DataFrame:
    """Construye la tabla de resumen multi-activo."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.summary_dict() for r in results.values()])
