"""Getting daily price bars, either from Alpaca or made up for testing."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["open", "high", "low", "close", "volume"]


def load_bars(symbol: str, start: str = "2016-01-01", refresh: bool = False) -> pd.DataFrame:
    """Daily bars for a symbol. Downloads from Alpaca the first time, then reads the saved CSV."""
    path = DATA_DIR / f"{symbol.upper()}.csv"
    if path.exists() and not refresh:
        return read_csv(path)

    bars = fetch_alpaca(symbol, start)
    DATA_DIR.mkdir(exist_ok=True)
    bars.to_csv(path)
    return bars


def read_csv(path: Path) -> pd.DataFrame:
    bars = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean(bars)


def fetch_alpaca(symbol: str, start: str) -> pd.DataFrame:
    # imported here so the rest of the project (and the tests) work without keys
    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key, secret = api_keys()
    client = StockHistoricalDataClient(key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
        # the free plan can't ask for the last 15 minutes of data
        end=datetime.now() - timedelta(minutes=20),
        # adjust for splits and dividends, otherwise a split looks like a crash
        adjustment=Adjustment.ALL,
    )
    df = client.get_stock_bars(request).df
    if df.empty:
        raise ValueError(f"Alpaca returned no data for {symbol}")
    df = df.reset_index(level="symbol", drop=True)
    df.index = pd.to_datetime(df.index).tz_convert("America/New_York").tz_localize(None).normalize()
    return clean(df)


def api_keys() -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Missing Alpaca keys. Copy .env.example to .env and add your paper trading keys, "
            "or run with --demo to use made-up prices."
        )
    return key, secret


def synthetic_bars(days: int = 2500, seed: int = 0, drift: float = 0.08, vol: float = 0.18) -> pd.DataFrame:
    """A random walk that looks like a stock but can't be predicted.

    Useful as a sanity check: if a model makes money on this, the backtest is cheating somewhere.
    """
    rng = np.random.default_rng(seed)
    daily = rng.normal(drift / 252 - vol**2 / 504, vol / np.sqrt(252), days)
    close = 100 * np.exp(np.cumsum(daily))
    open_ = close * np.exp(rng.normal(0, vol / np.sqrt(252) / 3, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, days)))
    volume = rng.lognormal(16, 0.3, days).round()
    index = pd.bdate_range("2015-01-02", periods=days)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def clean(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars[COLUMNS].astype(float).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    return bars.dropna()
