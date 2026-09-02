"""Descarga y normalización de datos de mercado vía yfinance."""
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

# Silenciar ruido de yfinance (tickers delisted/404) en logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)


@dataclass
class MarketData:
    """Contenedor de datos de mercado normalizados para un ticker."""

    ticker: str
    df: pd.DataFrame

    @property
    def last_price(self) -> float:
        if self.df is None or self.df.empty:
            return float("nan")
        return float(self.df["Close"].iloc[-1])

    @property
    def last_date(self):
        if self.df is None or self.df.empty:
            return None
        return self.df.index[-1]


class DataFetcher:
    """Descarga datos históricos y los normaliza en un DataFrame limpio."""

    # Columnas requeridas y su orden canónico
    COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust

    def fetch(
        self,
        ticker: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> MarketData:
        """Descarga datos de un ticker.

        Args:
            ticker: Símbolo de la acción (ej. "AAPL").
            period: Periodo de Yahoo ("2y", "5y", etc.).
            start: Fecha de inicio (YYYY-MM-DD).
            end: Fecha final (YYYY-MM-DD).
            interval: Granularidad ("1d" por defecto).
        """
        data = yf.download(
            ticker,
            period=period,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )

        if data.empty:
            raise ValueError(f"No se obtuvieron datos para el ticker '{ticker}'")

        df = self._normalize(data)
        return MarketData(ticker=ticker, df=df)

    def fetch_many(
        self,
        tickers: list[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> dict[str, MarketData]:
        """Descarga múltiples tickers secuencialmente.

        Los tickers que fallen se omiten (no se lanzan excepciones).
        """
        result: dict[str, MarketData] = {}
        for ticker in tickers:
            try:
                result[ticker] = self.fetch(
                    ticker,
                    period=period,
                    start=start,
                    end=end,
                    interval=interval,
                )
            except Exception:
                continue
        return result

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normaliza el DataFrame de yfinance.

        - Asegura las columnas requeridas.
        - Coloca index como datetime.
        - Ordena cronológicamente.
        - Elimina filas con precios nulos.
        """
        df = data.copy()

        # Si hay MultiIndex (multi-ticker) tomamos el primer símbolo
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Asegurar columnas
        missing = [c for c in self.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

        df = df[self.COLUMNS].copy()

        # Fechas
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Tipos numéricos y limpieza
        for col in self.COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remover filas sin precio de cierre válido
        df = df[df["Close"].notna()].copy()

        # Volumen >= 0
        df["Volume"] = df["Volume"].clip(lower=0)

        return df
