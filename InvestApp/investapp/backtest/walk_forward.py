"""Runner de walk-forward multi-activo x estrategia.

Descarga el total (calentamiento + operación), prepara los indicadores sobre
todo el histórico y opera día a día solo en la ventana de prueba, sin mirar el
futuro. Devuelve resultados comparables entre activos y estrategias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..data.ticker_list import TickerList
from ..strategies import (
    STRATEGIES,
    ConfluenceStrategy,
    HybridStrategy,
    MeanReversionStrategy,
    TrendFollowerStrategy,
)
from .engine import BacktestEngine, BacktestResult


@dataclass
class WalkForwardConfig:
    """Configuración del barrido walk-forward."""

    period: str = "5y"          # total descargado (calentamiento + operación)
    warmup: str = "2y"          # período de calentamiento (no se opera)
    initial_capital: float = 10_000.0
    fee_rate: float = 0.001

    @property
    def test_period(self) -> str:
        """Ventana operativa aproximada (total - calentamiento)."""
        return self.period


@dataclass
class WalkForwardSummary:
    """Resumen comparativo de una corrida."""

    ticker: str
    strategy: str
    retorno_total: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int
    decisiones: int
    signal_actual: str
    regime_actual: str
    mode_actual: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "estrategia": self.strategy,
            "retorno_total": round(self.retorno_total * 100, 2),
            "sharpe": round(self.sharpe, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "trades": self.trades,
            "decisiones": self.decisiones,
            "senal_actual": self.signal_actual,
            "regimen_actual": self.regime_actual,
            "modo_actual": self.mode_actual,
        }


@dataclass
class WalkForwardResult:
    """Resultado agregado del barrido."""

    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    results: list[BacktestResult] = field(default_factory=list)

    def keys(self) -> list[tuple[str, str]]:
        return [(r.ticker, r.strategy_name) for r in self.results]

    def get(self, ticker: str, strategy: str) -> Optional[BacktestResult]:
        for r in self.results:
            if r.ticker == ticker and r.strategy_name == strategy:
                return r
        return None

    @property
    def tickers(self) -> list[str]:
        return sorted({r.ticker for r in self.results})

    @property
    def strategies(self) -> list[str]:
        return sorted({r.strategy_name for r in self.results})

    def summary(self) -> list[WalkForwardSummary]:
        out = []
        for r in self.results:
            m = r.metrics
            signal_actual = ""
            if r.decisions:
                last = r.decisions[-1]
                if last.signal_compra:
                    signal_actual = "COMPRA"
                elif last.signal_venta:
                    signal_actual = "VENTA"
            regime_actual = r.decisions[-1].regime if r.decisions else ""
            mode_actual = r.decisions[-1].mode if r.decisions else ""
            out.append(
                WalkForwardSummary(
                    ticker=r.ticker,
                    strategy=r.strategy_name,
                    retorno_total=m.total_return,
                    sharpe=m.sharpe_ratio,
                    max_drawdown=m.max_drawdown,
                    win_rate=m.win_rate,
                    trades=len(r.trades),
                    decisiones=len(r.decisions),
                    signal_actual=signal_actual,
                    regime_actual=regime_actual,
                    mode_actual=mode_actual,
                )
            )
        return out

    def summary_df(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.summary()])

    def decisiones_csv(self, path: str) -> None:
        """Exporta el paso a paso (día a día) de todos los activos a un CSV."""
        frames = []
        for r in self.results:
            df = r.decisions_df()
            if df.empty:
                continue
            df.insert(0, "ticker", r.ticker)
            df.insert(1, "estrategia", r.strategy_name)
            frames.append(df)
        if not frames:
            return
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)


class WalkForwardRunner:
    """Corre el walk-forward sobre varios activos x estrategias."""

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        strategies: list[str] | None = None,
        engine: BacktestEngine | None = None,
    ):
        self.config = config or WalkForwardConfig()
        if strategies is None:
            strategies = STRATEGIES
        self.strategy_names = strategies
        self.engine = engine or BacktestEngine(
            initial_capital=self.config.initial_capital,
            fee_rate=self.config.fee_rate,
        )

    def _build_strategy(self, name: str):
        from ..strategies import (
            ConfluenceStrategy,
            HybridStrategy,
            MeanReversionStrategy,
            SupportResistanceStrategy,
            TrendFollowerStrategy,
        )

        if name == SupportResistanceStrategy.name:
            return SupportResistanceStrategy()
        if name == HybridStrategy.name:
            return HybridStrategy()
        if name == ConfluenceStrategy.name:
            return ConfluenceStrategy()
        if name == TrendFollowerStrategy.name:
            return TrendFollowerStrategy()
        if name == MeanReversionStrategy.name:
            return MeanReversionStrategy()
        raise ValueError(f"Estrategia desconocida: {name}")

    def run(
        self,
        tickers: list[str] | None = None,
        tickers_file: str | None = None,
        tqdm: bool = False,
    ) -> WalkForwardResult:
        if tickers is None:
            if tickers_file:
                tl = TickerList.load(tickers_file)
                tickers = tl.get()
            else:
                tickers = []
        if not tickers:
            raise ValueError("No hay tickers para escanear (pasalos por 'tickers' o tickers_file).")
        total = len(tickers) * len(self.strategy_names)

        results: list[BacktestResult] = []
        done = 0
        for ticker in tickers:
            for name in self.strategy_names:
                strategy = self._build_strategy(name)
                try:
                    res = self.engine.run(
                        ticker,
                        strategy,
                        period=self.config.period,
                        warmup=self.config.warmup,
                    )
                    results.append(res)
                except Exception as exc:  # datos faltantes, etc.
                    print(f"  [skip] {ticker}/{name}: {exc}")
                done += 1
                if tqdm:
                    print(f"[{done}/{total}] {ticker}/{name} OK")
        return WalkForwardResult(config=self.config, results=results)