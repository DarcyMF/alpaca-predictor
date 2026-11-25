"""Turning today's prediction into a paper trade on Alpaca.

This only ever talks to Alpaca's paper trading account (fake money). There is no way to
switch it to a live account from here, on purpose. By default it only prints what it
would do; pass --submit to actually send the paper order.
"""

from dataclasses import dataclass


@dataclass
class Plan:
    symbol: str
    prob_up: float
    holding: float  # shares currently held
    action: str  # "buy", "sell" or "hold"
    qty: float

    def describe(self) -> str:
        view = f"{self.prob_up:.1%} chance {self.symbol} closes higher tomorrow"
        if self.action == "hold":
            return f"{view}. No change (holding {self.holding:g} shares)."
        return f"{view}. Plan: {self.action} {self.qty:g} shares."


def plan(symbol: str, prob_up: float, holding: float, qty: float, threshold: float = 0.5) -> Plan:
    """Same rule as the backtest: hold the stock when the model leans up, cash when it doesn't."""
    if prob_up > threshold and holding <= 0:
        return Plan(symbol, prob_up, holding, "buy", qty)
    if prob_up <= threshold and holding > 0:
        return Plan(symbol, prob_up, holding, "sell", holding)
    return Plan(symbol, prob_up, holding, "hold", 0)


def paper_client():
    from alpaca.trading.client import TradingClient

    from .data import api_keys

    key, secret = api_keys()
    # paper=True is hard coded. This project never trades real money.
    return TradingClient(key, secret, paper=True)


def current_holding(client, symbol: str) -> float:
    from alpaca.common.exceptions import APIError

    try:
        return float(client.get_open_position(symbol).qty)
    except APIError:
        return 0.0  # no position


def submit(client, p: Plan) -> str:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if p.action == "hold":
        return "Nothing to send."
    side = OrderSide.BUY if p.action == "buy" else OrderSide.SELL
    # "day" market orders sent after the close fill at the next open
    order = client.submit_order(MarketOrderRequest(symbol=p.symbol, qty=p.qty, side=side, time_in_force=TimeInForce.DAY))
    return f"Paper order sent: {p.action} {p.qty:g} {p.symbol} (id {order.id}, status {order.status})"
