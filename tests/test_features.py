import numpy as np
import pytest
import pandas as pd

from predictor.data import synthetic_bars
from predictor.features import FEATURES, add_features, build, rsi


def test_features_never_look_ahead():
    """Changing tomorrow's prices must not change today's features."""
    bars = synthetic_bars(300, seed=1)
    before = add_features(bars)

    changed = bars.copy()
    changed.iloc[200:, :] *= 3  # wreck everything from day 200 on
    after = add_features(changed)

    pd.testing.assert_frame_equal(before[FEATURES].iloc[:200], after[FEATURES].iloc[:200])


def test_target_is_whether_tomorrow_closes_higher():
    idx = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": [10, 11, 10.5, 12], "volume": 1.0}, index=idx)
    df = build_without_dropping(bars)

    assert df["target"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(df["target"].iloc[-1])  # tomorrow isn't known yet
    assert df["next_ret"].iloc[0] == pytest.approx(0.1)


def test_build_drops_rows_without_enough_history():
    df = build(synthetic_bars(200))
    assert not df[FEATURES].isna().any().any()
    assert len(df) == 200 - 49  # the 50-day average is ready on day 50


def test_rsi_stays_between_0_and_100():
    values = rsi(synthetic_bars(500, seed=3)["close"]).dropna()
    assert values.between(0, 100).all()


def test_rsi_is_100_when_price_only_rises():
    close = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(close).iloc[-1] == 100


def build_without_dropping(bars):
    from predictor.features import add_target

    return add_target(bars)
