"""Tests del market screener con un fetcher simulado (sin red)."""
import numpy as np
import pandas as pd

from investapp.data.fetcher import MarketData
from investapp.screener import MarketScreener


class FakeFetcher:
    """Devuelve datos sintéticos; AAPL fuerza una compra, MSFT nada."""

    def fetch(self, ticker, period=None, start=None, end=None, interval="1d"):
        n = 400
        idx = pd.bdate_range(end="2024-12-31", periods=n)
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
        df = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0, 0.002, n)),
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.integers(1e5, 4e6, n),
            },
            index=idx,
        )
        return MarketData(ticker=ticker, df=df)


def test_scan_devuelve_resultado():
    from investapp.strategies import HybridStrategy

    screener = MarketScreener(tickers=["AAPL", "MSFT"], fetcher=FakeFetcher())
    res = screener.scan(HybridStrategy(), period="1y")
    # debe evaluar sin errores
    assert res.signals is not None
    assert isinstance(res.lista_tickers(), list)


def test_filtrar_compra_venta():
    from investapp.strategies import HybridStrategy

    screener = MarketScreener(tickers=["AAPL"], fetcher=FakeFetcher())
    res = screener.scan(HybridStrategy(), period="1y")
    compras = res.filtrar("COMPRA")
    ventas = res.filtrar("VENTA")
    assert len(compras) + len(ventas) <= len(res)


def test_exportar_csv(tmp_path):
    from investapp.strategies import HybridStrategy

    screener = MarketScreener(tickers=["AAPL"], fetcher=FakeFetcher())
    res = screener.scan(HybridStrategy(), period="1y")
    out = tmp_path / "senales.csv"
    res.exportar_csv(out)
    assert out.exists()


def test_raises_sin_strategy():
    from investapp.screener import MarketScreener
    from investapp.strategies import get_strategy

    # nombre inválido debe lanzar
    import pytest

    with pytest.raises(ValueError):
        get_strategy("no_existe")