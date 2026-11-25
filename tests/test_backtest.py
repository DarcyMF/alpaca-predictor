import pandas as pd
import pytest

from predictor import backtest


def days(next_rets):
    idx = pd.bdate_range("2024-01-01", periods=len(next_rets))
    s = pd.Series(next_rets, index=idx, dtype=float)
    return pd.DataFrame({"next_ret": s, "target": (s > 0).astype(float)})


def test_holding_every_day_matches_buy_and_hold_minus_one_trade():
    df = days([0.01, -0.02, 0.03])
    prob = pd.Series(0.9, index=df.index, name="test")

    r = backtest.run(df, prob, cost_bps=10)

    expected = (1.01 - 0.001) * 0.98 * 1.03 - 1
    assert r.total_return == pytest.approx(expected)
    assert r.trades == 1
    assert r.time_in_market == 1


def test_cash_earns_nothing_and_costs_nothing():
    df = days([0.05, -0.05, 0.05])
    r = backtest.run(df, pd.Series(0.1, index=df.index, name="cash"))

    assert r.total_return == 0
    assert r.trades == 0


def test_every_switch_pays_the_cost():
    df = days([0.0, 0.0, 0.0, 0.0])
    prob = pd.Series([0.9, 0.1, 0.9, 0.1], index=df.index, name="flip")

    r = backtest.run(df, prob, cost_bps=100)

    assert r.trades == 4  # buy, sell, buy, sell
    assert r.total_return == pytest.approx(0.99**4 - 1)


def test_accuracy_counts_right_calls():
    df = days([0.01, -0.01, 0.01, -0.01])
    prob = pd.Series([0.9, 0.9, 0.1, 0.1], index=df.index, name="half")

    assert backtest.run(df, prob).accuracy == 0.5


def test_last_day_without_an_answer_is_skipped():
    df = days([0.01, 0.02])
    df.loc[df.index[-1], ["next_ret", "target"]] = float("nan")
    r = backtest.run(df, pd.Series(0.9, index=df.index, name="x"), cost_bps=0)

    assert len(r.returns) == 1
    assert r.total_return == pytest.approx(0.01)


def test_drawdown():
    df = days([0.10, -0.50, 0.20])
    r = backtest.buy_and_hold(df, df.index)
    assert r.max_drawdown == pytest.approx(-0.5)
