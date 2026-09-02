"""Tests de indicadores técnicos."""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def df():
    n = 300
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    close = np.linspace(100, 130, n)
    open_ = close * 0.999
    high = close * 1.01
    low = close * 0.99
    vol = np.full(n, 1_000_000)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def test_sma(df):
    from investapp.indicators.technical import sma

    s = sma(df["Close"], 20)
    assert s.iloc[19] == pytest.approx(df["Close"].iloc[:20].mean())
    assert s.iloc[0] != s.iloc[0]  # NaN


def test_rsi_rango(df):
    from investapp.indicators.technical import rsi

    r = rsi(df["Close"], 14)
    valid = r.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_todos_up():
    from investapp.indicators.technical import rsi

    s = pd.Series(np.linspace(10, 50, 40))
    r = rsi(s, 14).dropna()
    assert r.iloc[-1] == pytest.approx(100.0, abs=0.5)


def test_macd_columnas(df):
    from investapp.indicators.technical import macd

    m = macd(df["Close"])
    assert {"macd", "signal", "hist"}.issubset(m.columns)


def test_bollinger_bounds(df):
    from investapp.indicators.volatility import bollinger_bands

    bb = bollinger_bands(df["Close"], 20, 2.0)
    ok = bb.dropna()
    assert (ok["lower"] <= ok["mid"]).all()
    assert (ok["mid"] <= ok["upper"]).all()


def test_atr_positivo(df):
    from investapp.indicators.technical import atr

    a = atr(df, 14).dropna()
    assert (a > 0).all()


def test_adx_y_di(df):
    from investapp.indicators.technical import adx

    a = adx(df).dropna()
    assert {"adx", "plus_di", "minus_di"}.issubset(a.columns)
    assert ((a >= 0) & (a <= 100)).all().all()


def test_stochastic(df):
    from investapp.indicators.technical import stochastic

    st = stochastic(df).dropna()
    assert ((st >= 0) & (st <= 100)).all().all()


def test_obv(df):
    from investapp.indicators.volume import obv

    o = obv(df)
    assert len(o) == len(df)


def test_relative_volume_uno(df):
    from investapp.indicators.volume import relative_volume

    rv = relative_volume(df, 20).dropna()
    assert rv.iloc[-1] == pytest.approx(1.0)