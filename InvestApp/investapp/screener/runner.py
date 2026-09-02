"""Runner del Market Screener: barre muchos activos y devuelve los que
están en momento de compra o de venta según la estrategia elegida."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.fetcher import DataFetcher, MarketData
from ..data.ticker_list import TickerList
from ..strategies.base import Strategy
from ..strategies import get_strategy


@dataclass
class AssetSignal:
    """Señal detectada para un activo."""

    ticker: str
    senal: str  # "COMPRA" | "VENTA" | "NEUTRO"
    estrategia: str
    fecha: Optional[pd.Timestamp] = None
    precio: float = 0.0
    entrada: float = 0.0
    stop_loss: float = 0.0
    take_profit: Optional[float] = None
    rr: float = 0.0
    strength: float = 0.0
    regime: str = ""
    reasons: list[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    df: Optional[pd.DataFrame] = None  # features para gráficos del reporte
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "senal": self.senal,
            "estrategia": self.estrategia,
            "fecha": str(self.fecha.date()) if self.fecha is not None else "",
            "precio": self.precio,
            "entrada": self.entrada,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit or "",
            "rr": self.rr,
            "strength": self.strength,
            "regime": self.regime,
            "reasons": " | ".join(self.reasons),
        }


@dataclass
class ScanResult:
    """Resultado de un barrido sobre la lista de tickers."""

    signals: list[AssetSignal] = field(default_factory=list)
    strategy_name: str = ""

    def filtrar(self, senal: str = "COMPRA") -> "ScanResult":
        """Filtra por señal: COMPRA / VENTA / NEUTRO."""
        return ScanResult(
            signals=[s for s in self.signals if s.senal.upper() == senal.upper()],
            strategy_name=self.strategy_name,
        )

    def lista_tickers(self) -> list[str]:
        return [s.ticker for s in self.signals]

    def intersectar(self, otro: "ScanResult") -> "ScanResult":
        """Devuelve los activos señalados en ambos resultados."""
        otros_tickers = set(otro.lista_tickers())
        return ScanResult(
            signals=[s for s in self.signals if s.ticker in otros_tickers],
            strategy_name=f"{self.strategy_name} ∩ {otro.strategy_name}",
        )

    def exportar_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([s.to_dict() for s in self.signals]).to_csv(
            path, index=False, encoding="utf-8-sig"
        )

    def exportar_html(self, path: str | Path, title: str | None = None) -> None:
        """Exporta un reporte visual HTML con tablas y gráficos embebidos."""
        from .report_html import build_html

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = build_html(self, title=title)
        path.write_text(html, encoding="utf-8")

    def mostrar_cli(self, top: int | None = None) -> None:
        """Muestra la tabla de señales en terminal."""
        from .report_cli import print_table

        print_table(self, top=top)

    def __len__(self) -> int:
        return len(self.signals)


class MarketScreener:
    """Barre una lista de tickers con una estrategia y reporta señales."""

    def __init__(
        self,
        tickers_file: str | Path | None = None,
        tickers: Optional[list[str]] = None,
        fetcher: Optional[DataFetcher] = None,
        min_bars: int = 60,
        verbose: bool = False,
    ):
        self.ticker_list = TickerList()
        if tickers_file:
            self.ticker_list = TickerList.load(tickers_file)
        if tickers:
            self.ticker_list.add_many(tickers)
        self.fetcher = fetcher or DataFetcher()
        self.min_bars = min_bars
        self.verbose = verbose

    def add_tickers(self, tickers: list[str]) -> None:
        self.ticker_list.add_many(tickers)

    def scan(
        self,
        strategy: Strategy | str,
        period: str = "2y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> ScanResult:
        """Ejecuta el barrido sobre todos los tickers configurados."""
        strat = strategy if isinstance(strategy, Strategy) else get_strategy(strategy, risk=None)
        results: list[AssetSignal] = []

        tickers = self.ticker_list.get()
        if not tickers:
            print("Advertencia: no hay tickers configurados para el barrido.")

        for i, ticker in enumerate(tickers, 1):
            if self.verbose:
                print(f"[{i}/{len(tickers)}] Escaneando {ticker}...")
            try:
                md = self.fetcher.fetch(ticker, period=period, start=start, end=end)
                signal = self._evaluate(ticker, md, strat)
                if signal is not None:
                    results.append(signal)
                elif self.verbose:
                    print(f"  {ticker}: sin señal.")
            except Exception as exc:
                results.append(
                    AssetSignal(
                        ticker=ticker,
                        senal="NEUTRO",
                        estrategia=strat.name,
                        error=str(exc),
                    )
                )
                if self.verbose:
                    print(f"  {ticker}: error ({exc})")

        return ScanResult(signals=results, strategy_name=strat.name)

    def _evaluate(
        self, ticker: str, md: MarketData, strategy: Strategy
    ) -> Optional[AssetSignal]:
        df = md.df
        if len(df) < self.min_bars:
            return None

        # Recalcular features para este ticker
        df = strategy.prepare(df.copy())
        df.attrs["ticker"] = ticker

        ev = strategy.evaluate(-1)
        price = float(df["Close"].iloc[-1])
        indicators = ev.indicators

        sig = ev.signal
        if sig is not None:
            rr = 0.0
            if sig.take_profit is not None and sig.stop_loss > 0 and sig.entry > sig.stop_loss:
                rr = (sig.take_profit - sig.entry) / (sig.entry - sig.stop_loss)
            return AssetSignal(
                ticker=ticker,
                senal="COMPRA",
                estrategia=strategy.name,
                fecha=df.index[-1],
                precio=price,
                entrada=sig.entry,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                rr=round(rr, 2),
                strength=sig.strength,
                regime=ev.context.get("regime", ""),
                reasons=sig.reasons,
                indicators=indicators,
                df=df,
            )

        if ev.should_exit and ev.exit_fresh:
            return AssetSignal(
                ticker=ticker,
                senal="VENTA",
                estrategia=strategy.name,
                fecha=df.index[-1],
                precio=price,
                regime=ev.context.get("regime", ""),
                reasons=ev.context.get("exit_reasons", []),
                indicators=indicators,
                df=df,
            )

        return None