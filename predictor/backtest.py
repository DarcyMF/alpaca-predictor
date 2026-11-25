"""Turning predictions into a trading rule and measuring how it would have done.

Rule: at the close, if the model says tomorrow is more likely up than the threshold,
hold the stock for tomorrow, otherwise hold cash. Every buy or sell pays a cost.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    name: str
    returns: pd.Series  # daily strategy returns after costs
    accuracy: float  # how often the up/down call was right
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    time_in_market: float
    trades: int

    def row(self) -> dict:
        return {
            "strategy": self.name,
            "accuracy": self.accuracy,
            "total return": self.total_return,
            "per year": self.cagr,
            "sharpe": self.sharpe,
            "max drawdown": self.max_drawdown,
            "time in market": self.time_in_market,
            "trades": self.trades,
        }


def run(df: pd.DataFrame, prob_up: pd.Series, threshold: float = 0.5, cost_bps: float = 5.0) -> Result:
    """Backtest one set of predictions. cost_bps is the cost per buy or sell, in hundredths of a percent."""
    days = df.loc[prob_up.index].dropna(subset=["next_ret"])
    prob = prob_up.loc[days.index]

    position = (prob > threshold).astype(float)
    # a trade happens whenever the position changes, including the first buy
    trades = position.diff().abs().fillna(position.iloc[0])
    returns = position * days["next_ret"] - trades * cost_bps / 10_000

    called_up = prob > 0.5
    went_up = days["target"] == 1
    accuracy = float((called_up == went_up).mean())
    return summarise(str(prob_up.name), returns, accuracy, position.mean(), int(trades.sum()))


def buy_and_hold(df: pd.DataFrame, index: pd.Index) -> Result:
    days = df.loc[index].dropna(subset=["next_ret"])
    accuracy = float((days["target"] == 1).mean())
    return summarise("buy_and_hold", days["next_ret"], accuracy, 1.0, 1)


def summarise(name: str, returns: pd.Series, accuracy: float, in_market: float, trades: int) -> Result:
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    drawdown = float((equity / equity.cummax() - 1).min())
    return Result(name, returns, accuracy, total, cagr, sharpe, drawdown, float(in_market), trades)


def table(results: list[Result]) -> pd.DataFrame:
    t = pd.DataFrame([r.row() for r in results]).set_index("strategy")
    pct = ["accuracy", "total return", "per year", "max drawdown", "time in market"]
    t[pct] = (t[pct] * 100).round(1)
    t["sharpe"] = t["sharpe"].round(2)
    return t
