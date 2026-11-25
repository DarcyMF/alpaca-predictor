"""Turning prices into the inputs the model learns from.

Every feature on day t only uses prices up to and including day t. The target is whether
day t+1 closes higher, so the model is predicting tomorrow from what it knows tonight.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "sma_gap_10",
    "sma_gap_50",
    "rsi_14",
    "macd_hist",
    "volume_z",
    "range",
    "day_of_week",
]


def build(bars: pd.DataFrame) -> pd.DataFrame:
    """Features and target for each day. Rows without enough history are dropped."""
    df = add_features(bars)
    df = add_target(df)
    return df.dropna(subset=FEATURES)


def add_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    close = df["close"]
    ret = close.pct_change()

    df["ret_1"] = ret
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)
    df["vol_10"] = ret.rolling(10).std()
    df["vol_20"] = ret.rolling(20).std()
    df["sma_gap_10"] = close / close.rolling(10).mean() - 1
    df["sma_gap_50"] = close / close.rolling(50).mean() - 1
    df["rsi_14"] = rsi(close, 14)
    df["macd_hist"] = macd_histogram(close) / close
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["range"] = (df["high"] - df["low"]) / close
    df["day_of_week"] = df.index.dayofweek
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_ret"] = df["close"].shift(-1) / df["close"] - 1
    # 1 if tomorrow closes higher, NaN on the last day because we don't know yet
    df["target"] = (df["next_ret"] > 0).astype(float).where(df["next_ret"].notna())
    return df


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index (Wilder's smoothing), 0 to 100."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(gain.notna())


def macd_histogram(close: pd.Series) -> pd.Series:
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd - signal
