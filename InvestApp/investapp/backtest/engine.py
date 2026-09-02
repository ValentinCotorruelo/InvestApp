"""Motor de backtesting de estrategias long-only sobre datos diarios."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher, MarketData
from ..risk.position_sizing import PositionSizingParams, compute_position_size
from ..risk.risk_management import (
    RiskManagementParams,
    update_trailing_stop,
)
from ..strategies.base import Strategy
from ..utils.metrics import PerformanceMetrics, compute_metrics


def _offset_days(offset: str) -> int:
    """Convierte '3y', '18m', '45d', '2w' a días (aproximación calendario)."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(y|m|w|d)\s*$", offset.lower())
    if not m:
        raise ValueError(f"Offset inválido: {offset!r} (usá ej. '2y', '18m', '45d').")
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "y":
        return int(round(value * 365))
    if unit == "m":
        return int(round(value * 30))
    if unit == "w":
        return int(round(value * 7))
    return int(round(value))


@dataclass
class DayDecision:
    """Registro de la decisión de un día en la ventana de operación.

    Es el "paso a paso": qué indicadores había, si había señal de compra o
    venta, y los niveles de entrada/stop/take vigentes. Solo usa datos hasta
    ese día.
    """

    date: pd.Timestamp
    price: float
    action: str  # "entry" | "exit" | "hold"
    signal_compra: bool = False
    signal_venta: bool = False
    had_position: bool = False
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing: Optional[float] = None
    reason: str = ""
    regime: str = "unknown"
    mode: str = ""
    indicators: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "date": self.date,
            "price": round(self.price, 4),
            "action": self.action,
            "signal_compra": self.signal_compra,
            "signal_venta": self.signal_venta,
            "had_position": self.had_position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing": self.trailing,
            "reason": self.reason,
            "regime": self.regime,
            "mode": self.mode,
        }
        d.update(self.indicators)
        return d


@dataclass
class Trade:
    """Una operación completada por el backtest."""

    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    reason: str = ""
    hold_bars: int = 0
    status: str = "closed"  # "closed" | "open" (valuada al cierre del último día)

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "shares": self.shares,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "reason": self.reason,
            "hold_bars": self.hold_bars,
            "status": self.status,
        }


@dataclass
class BacktestResult:
    """Resultado de una corrida de backtest."""

    ticker: str
    strategy_name: str
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    events: list[dict] = field(default_factory=list)
    decisions: list[DayDecision] = field(default_factory=list)
    test_start: Optional[pd.Timestamp] = None
    df: Optional[pd.DataFrame] = None  # OHLCV (+features) de la ventana de operación

    def __post_init__(self):
        trades = [t.to_dict() for t in self.trades]
        self.metrics = compute_metrics(self.equity, trades)
        self.metrics.trades = trades

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def decisions_df(self) -> pd.DataFrame:
        if not self.decisions:
            return pd.DataFrame(columns=["date", "price", "action"])
        return pd.DataFrame([d.to_dict() for d in self.decisions])

    def report(self) -> None:
        """Imprime el resumen de métricas."""
        header = f"Backtest {self.ticker} | Estrategia: {self.strategy_name}"
        print("=" * len(header))
        print(header)
        print("=" * len(header))
        print(self.metrics.report())
        print(f"Trades: {len(self.trades)}")
        if self.trades:
            print("\nÚltimos 5 trades:")
            print(self.trades_df().tail(5).to_string(index=True))

    def list_trades(self, limit: int | None = None) -> pd.DataFrame:
        df = self.trades_df()
        if limit is not None:
            df = df.head(limit)
        return df

    def plot(self, figsize=(14, 7), title: str | None = None, filename: str | None = None):
        """Genera un gráfico de la curva de equity y del precio con entradas/salidas."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib no instalado; no se puede graficar.")
            return

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Equity
        axes[0].plot(self.equity.index, self.equity.values, label="Equity", lw=1.5)
        axes[0].set_title(title or f"Backtest {self.ticker} | {self.strategy_name}")
        axes[0].set_ylabel("Equity ($)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Drawdown
        dd = self.equity / self.equity.cummax() - 1
        axes[1].fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.4)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=120, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


class BacktestEngine:
    """Ejecuta una estrategia sobre un activo paso a paso, día por día."""

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        sizing: Optional[PositionSizingParams] = None,
        fee_rate: float = 0.001,  # 0.1% por operación (ida y vuelta)
        fill: str = "open",  # "open" | "close": donde ejecutar señales extremas
    ):
        if fill not in ("open", "close", "next_open"):
            raise ValueError(f"fill inválido: {fill!r} (usá 'open', 'close' o 'next_open').")
        self.initial_capital = initial_capital
        self.sizing = sizing or PositionSizingParams()
        self.fee_rate = fee_rate
        self.fill = fill
        self.fetcher = DataFetcher()

    def run(
        self,
        asset: str | MarketData | pd.DataFrame,
        strategy: Strategy,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "3y",
        warmup: Optional[str] = None,
        cooldown_days: Optional[int] = None,
    ) -> BacktestResult:
        """Corre el backtest.

        Args:
            asset: ticker ("AAPL"), MarketData o DataFrame con OHLCV.
            strategy: instancia de estrategia.
            start/end/period: ventana temporal de datos.
            warmup: período de calentamiento (ej. "2y"). Los primeros
                'warmup' se usan solo para que los indicadores y niveles
                "maduren"; la operación (compras/ventas) empieza después.
            cooldown_days: días de espera tras un stop-loss (default: de la estrategia).
        """
        df = self._resolve_df(asset, start, end, period, warmup)
        if df is None or len(df) < 60:
            raise ValueError("Datos insuficientes para backtest.")

        # Precomputar features
        df = strategy.prepare(df)
        risk: RiskManagementParams = strategy.risk
        cooldown = cooldown_days if cooldown_days is not None else risk.cooldown_days

        # ---- Ventana de operación (walk-forward) ----
        warmup_idx = 0
        test_start_ts = None
        if warmup is not None:
            if start is not None:
                test_start_ts = pd.Timestamp(start)
            else:
                warmup_days = _offset_days(warmup)
                test_days = max(_offset_days(period) - warmup_days, 1)
                test_start_ts = df.index[-1] - pd.Timedelta(days=test_days)
            warmup_idx = int(df.index.searchsorted(test_start_ts, side="left"))
            if warmup_idx < max(250, strategy.min_bars_required + 5):
                raise ValueError(
                    f"Período de calentamiento insuficiente: {warmup_idx} barras "
                    f"antes de {test_start_ts.date()}. Aumentá warmup o period."
                )

        closes = df["Close"].to_numpy()
        opens = df["Open"].to_numpy()
        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()
        atr_arr = df["ATR"].fillna(0.0).to_numpy()
        index = df.index
        n = len(df)

        cash = self.initial_capital
        shares = 0
        entry_price = 0.0
        last_entry_price = 0.0
        entry_date = None
        stop_loss = 0.0
        take_profit: float | None = None
        trailing: float | None = None
        trailing_mult = risk.trailing_atr_multiplier
        entry_index = -1
        hold_bars = 0
        last_stop_date_offset = -10**9  # cooldown

        # Órdenes pendientes para el modo "next_open" (señal en t, ejecución en t+1)
        pending_buy_reason = ""
        pending_sell_reason = ""

        equity = pd.Series(index=index, dtype=float)
        trades: list[Trade] = []
        events: list[dict] = []
        decisions: list[DayDecision] = []

        def _open_position(i, fill_price, sig) -> bool:
            """Abre una posición long. Devuelve True si se abrió."""
            nonlocal shares, cash, entry_date, stop_loss, take_profit, trailing
            nonlocal trailing_mult, hold_bars, entry_index, last_entry_price, entry_price
            if sig is None or sig.entry <= 0 or sig.stop_loss <= 0:
                return False
            if sig.stop_loss >= fill_price:
                return False
            shares_calc, notional, _ = compute_position_size(
                cash, fill_price, sig.stop_loss, self.sizing
            )
            if shares_calc <= 0:
                return False
            cost = shares_calc * fill_price
            fees = cost * self.fee_rate
            if cost + fees > cash:
                return False
            shares = shares_calc
            cash -= cost + fees
            entry_date = date
            entry_price = fill_price
            last_entry_price = fill_price
            stop_loss = sig.stop_loss
            take_profit = sig.take_profit
            if take_profit is not None and take_profit <= fill_price:
                take_profit = None
            trailing_mult = (
                sig.meta.get("trailing_mult", risk.trailing_atr_multiplier)
                or risk.trailing_atr_multiplier
            )
            trailing = fill_price - (
                float(atr_arr[i]) * trailing_mult if float(atr_arr[i]) > 0 else 0.0
            )
            entry_index = i
            hold_bars = 0
            events.append(
                {
                    "date": date,
                    "type": "entry",
                    "price": fill_price,
                    "strength": sig.strength,
                    "reasons": sig.reasons,
                }
            )
            return True

        def _close_position(exit_price, reason, i):
            """Cierra la posición abierta."""
            nonlocal shares, cash, entry_price, stop_loss, take_profit, trailing, hold_bars
            proceeds = exit_price * shares
            fees = proceeds * self.fee_rate
            cash += proceeds - fees
            pnl = (exit_price - entry_price) * shares
            trades.append(
                Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=date,
                    exit_price=exit_price,
                    shares=shares,
                    pnl=pnl,
                    return_pct=(exit_price / entry_price - 1) if entry_price else 0.0,
                    reason=reason,
                    hold_bars=hold_bars,
                )
            )
            events.append(
                {"date": date, "type": "exit", "price": exit_price, "reason": reason}
            )
            if reason == "stop_loss":
                nonlocal last_stop_date_offset
                last_stop_date_offset = i
            shares = 0
            entry_price = 0.0
            stop_loss = 0.0
            take_profit = None
            trailing = None
            hold_bars = 0

        # Variable de llenado pendiente (modo "next_open"): señal de compra/venta
        # generada en el día t que se ejecuta al OPEN de t+1.
        use_next_open = self.fill == "next_open"
        pending_buy_signal = None
        pending_sell = False

        entered_today = False
        exited_today = False

        for i in range(n):
            date = index[i]
            close = float(closes[i])
            low = float(lows[i])
            high = float(highs[i])
            in_test = i >= warmup_idx
            entered_today = False
            exited_today = False

            # ---- 1) Ejecutar órdenes pendientes del día anterior (next_open) ----
            if use_next_open:
                open_i = float(opens[i])

                # Venta pendiente por señal de estrategia → cerrar al open
                if pending_sell and shares > 0:
                    if in_test:
                        _close_position(open_i, "señal_salida", i)
                        exited_today = True
                    else:
                        shares = 0  # fuera de ventana: no se registra
                    pending_sell = False

                # Compra pendiente por señal → abrir al open
                if pending_buy_signal is not None:
                    sig = pending_buy_signal
                    pending_buy_signal = None
                    if (
                        shares == 0
                        and in_test
                        and (i - last_stop_date_offset) >= cooldown
                    ):
                        if _open_position(i, open_i, sig):
                            entered_today = True

            # ---- 2) Evaluar la estrategia del día (usando datos hasta t) ----
            ev = strategy.evaluate(i)

            # Snapshot para el registro día a día
            if in_test:
                decision = DayDecision(
                    date=date,
                    price=close,
                    action="hold",
                    signal_compra=ev.signal is not None,
                    signal_venta=ev.should_exit,
                    had_position=shares > 0,
                    entry_price=entry_price if shares > 0 else None,
                    stop_loss=stop_loss if shares > 0 else (ev.signal.stop_loss if ev.signal else None),
                    take_profit=take_profit if shares > 0 else (ev.signal.take_profit if ev.signal else None),
                    trailing=trailing if shares > 0 else None,
                    reason="; ".join(ev.context.get("exit_reasons", [])) if ev.should_exit else "",
                    regime=ev.context.get("regime", "unknown"),
                    mode=ev.context.get("mode", ""),
                    indicators=dict(ev.indicators),
                )

            # ---- 3) Gestión de la posición vigente (stops/takes intrabarra) ----
            exit_reason_today = ""
            if shares > 0:
                if low <= stop_loss:
                    exit_reason_today = "stop_loss"
                elif take_profit is not None and high >= take_profit:
                    exit_reason_today = "take_profit"
                elif trailing is not None and low <= trailing:
                    exit_reason_today = "trailing_stop"

                if not use_next_open and exit_reason_today == "" and ev.should_exit:
                    # Señal de salida ejecutada al cierre del mismo día (modos open/close)
                    exit_reason_today = (
                        "; ".join(ev.context.get("exit_reasons", [])) or "señal_salida"
                    )

                # En next_open, las señales de salida NO se ejecutan intrabarra:
                # pasan a ser orden pendiente para el open siguiente. Solo stops/takes
                # intrabarra se resuelven acá.
                if exit_reason_today:
                    exit_price = {
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "trailing_stop": trailing,
                    }.get(
                        exit_reason_today,
                        close if not use_next_open else None,
                    )
                    if exit_price is not None and in_test:
                        _close_position(exit_price, exit_reason_today, i)
                    elif exit_price is not None:
                        shares = 0
                elif use_next_open and ev.should_exit and shares > 0:
                    pending_sell = True

            # ---- 4) UPDATE trailing (aún en posición) ----
            if shares > 0 and trailing_mult is not None:
                atr_val = float(atr_arr[i])
                if atr_val > 0:
                    trailing = update_trailing_stop(
                        trailing, close, atr_val, trailing_mult
                    )

            # ---- 5) Entrada: en modos open/close se ejecuta hoy; next_open difiere ----
            if not use_next_open:
                if (
                    shares == 0
                    and in_test
                    and (i - last_stop_date_offset) >= cooldown
                    and ev.signal is not None
                ):
                    entry_px = close if self.fill == "close" else ev.signal.entry
                    if _open_position(i, entry_px, ev.signal):
                        entered_today = True
            else:
                # next_open: si hay señal de compra elegible hoy, dejar pendiente
                # para ejecutarla al open de mañana (no acumular si ya hay señal).
                if (
                    pending_buy_signal is None
                    and shares == 0
                    and ev.signal is not None
                ):
                    pending_buy_signal = ev.signal

            if shares > 0:
                hold_bars += 1

            # ---- Equity diario ----
            equity.iloc[i] = cash + shares * close

            # ---- Registrar la decisión del día (ventana de operación) ----
            if in_test:
                if exited_today:
                    decision.action = "exit"
                    decision.reason = exit_reason_today or "señal_salida"
                elif entered_today:
                    decision.action = "entry"
                    decision.entry_price = last_entry_price
                    decision.stop_loss = stop_loss
                    decision.take_profit = take_profit
                    decision.trailing = trailing
                elif exit_reason_today:
                    decision.action = "exit"
                    decision.reason = exit_reason_today
                decisions.append(decision)

        # La curva de equity y métricas cubren solo la ventana de operación
        if warmup_idx > 0:
            equity = equity.iloc[warmup_idx:]
            df_test = df.iloc[warmup_idx:]
        else:
            df_test = df

        # Posición abierta al final del período: valuarla al cierre del último
        # día disponible y marcarla explícitamente como "open".
        if shares > 0 and entry_date is not None:
            open_mark = float(closes[-1])
            trades.append(
                Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=index[-1],
                    exit_price=open_mark,
                    shares=shares,
                    pnl=(open_mark - entry_price) * shares,
                    return_pct=(open_mark / entry_price - 1) if entry_price else 0.0,
                    reason="posicion_abierta",
                    hold_bars=hold_bars,
                    status="open",
                )
            )
            events.append(
                {
                    "date": index[-1],
                    "type": "open",
                    "price": open_mark,
                    "reason": "posicion_abierta",
                }
            )

        return BacktestResult(
            ticker=self._ticker_name(asset),
            strategy_name=strategy.name,
            equity=equity,
            trades=trades,
            metrics=PerformanceMetrics(),
            events=events,
            decisions=decisions,
            test_start=test_start_ts,
            df=df_test,
        )

    # ---- Helpers ----

    def _resolve_df(self, asset, start, end, period, warmup=None):
        """Devuelve un DataFrame normalizado de OHLCV.

        Con warmup y un 'start' explícito, descarga datos desde antes para que
        los indicadores maduren antes de la ventana de operación.
        """
        if isinstance(asset, MarketData):
            return asset.df
        if isinstance(asset, pd.DataFrame):
            df = asset.copy()
            return df
        # string ticker
        fetch_start = start
        if start is not None and warmup is not None:
            fetch_start = (
                pd.Timestamp(start) - pd.Timedelta(days=_offset_days(warmup))
            ).date().isoformat()
        md = self.fetcher.fetch(
            asset, period=period, start=fetch_start, end=end
        )
        return md.df

    def _ticker_name(self, asset) -> str:
        if isinstance(asset, MarketData):
            return asset.ticker
        if isinstance(asset, pd.DataFrame):
            return asset.attrs.get("ticker", "unknown")
        return str(asset)