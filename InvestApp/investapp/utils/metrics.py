"""Métricas de rendimiento para backtest y reportes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Conjunto de métricas financieras de una serie de retornos/equity."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    trades: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
        }

    def report(self) -> str:
        lines = [
            f"{'Métrica':<28} {'Valor':>12}",
            "-" * 42,
            f"{'Retorno total':<28} {self.total_return:>11.2%}",
            f"{'Retorno anualizado':<28} {self.annualized_return:>11.2%}",
            f"{'Volatilidad anualizada':<28} {self.annualized_volatility:>11.2%}",
            f"{'Sharpe ratio':<28} {self.sharpe_ratio:>12.2f}",
            f"{'Sortino ratio':<28} {self.sortino_ratio:>12.2f}",
            f"{'Máximo drawdown':<28} {self.max_drawdown:>11.2%}",
            f"{'Win rate':<28} {self.win_rate:>11.2%}",
            f"{'Número de trades':<28} {self.num_trades:>12}",
            f"{'Profit factor':<28} {self.profit_factor:>12.2f}",
            f"{'Expectancy (por trade)':<28} {self.expectancy:>12.4f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_curve: pd.Series,
    trades: list | None = None,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Calcula métricas a partir de una curva de equity y lista de trades."""
    if equity_curve is None or len(equity_curve) < 2:
        return PerformanceMetrics()

    rets = equity_curve.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    n = len(equity_curve)
    years = n / periods_per_year
    annualized_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 and total_return > -1 else -1.0
    annualized_vol = float(rets.std() * np.sqrt(periods_per_year)) if len(rets) > 1 else 0.0

    sharpe = 0.0
    sortino = 0.0
    if annualized_vol > 0:
        sharpe = float(annualized_return / annualized_vol)
    downside = rets[rets < 0].std() * np.sqrt(periods_per_year)
    if downside > 0:
        sortino = float(annualized_return / downside)

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = float(drawdown.min())

    # Métricas por trade
    trades = trades or []
    num_trades = len(trades)
    win_rate = 0.0
    profit_factor = 0.0
    expectancy = 0.0
    if num_trades > 0:
        pct = [t.get("return_pct", 0.0) for t in trades]
        wins = [p for p in pct if p > 0]
        losses = [p for p in pct if p <= 0]
        win_rate = float(len(wins) / num_trades)
        gross_win = sum(p * (1 + 0) for p in wins)
        gross_win_val = sum(wins)
        gross_loss_val = -sum(losses)
        if gross_loss_val > 0:
            profit_factor = float(gross_win_val / gross_loss_val)
        expectancy = float(np.mean(pct)) if pct else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=num_trades,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trades=trades,
    )