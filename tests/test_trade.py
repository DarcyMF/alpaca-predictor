import sys
import types

from predictor import trade


def test_buys_when_model_leans_up_and_not_holding():
    p = trade.plan("SPY", 0.56, holding=0, qty=2)
    assert (p.action, p.qty) == ("buy", 2)


def test_sells_everything_when_model_leans_down():
    p = trade.plan("SPY", 0.44, holding=3, qty=2)
    assert (p.action, p.qty) == ("sell", 3)


def test_holds_otherwise():
    assert trade.plan("SPY", 0.6, holding=1, qty=1).action == "hold"
    assert trade.plan("SPY", 0.4, holding=0, qty=1).action == "hold"


def test_client_is_always_paper(monkeypatch):
    """The only account this project can reach is the paper (fake money) one."""
    created = {}

    class FakeClient:
        def __init__(self, key, secret, paper):
            created["paper"] = paper

    fake = types.ModuleType("alpaca.trading.client")
    fake.TradingClient = FakeClient
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", fake)
    monkeypatch.setattr("predictor.data.api_keys", lambda: ("k", "s"))

    trade.paper_client()
    assert created["paper"] is True


def test_hold_sends_nothing():
    class Explodes:
        def submit_order(self, *_):
            raise AssertionError("should not send an order")

    assert trade.submit(Explodes(), trade.plan("SPY", 0.6, 1, 1)) == "Nothing to send."
