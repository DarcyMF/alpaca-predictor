"""Command line: python -m predictor <command>

  backtest SPY            walk-forward backtest with a chart in reports/
  backtest --demo         same thing on made-up prices, no keys needed
  signal SPY              today's prediction and what the paper account would do
  signal SPY --submit     actually send that order to the paper account
"""

import argparse

from . import backtest, model, report, trade
from .data import load_bars, synthetic_bars
from .features import build


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="predictor", description="Next-day stock direction model with Alpaca data")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="walk-forward backtest")
    bt.add_argument("symbol", nargs="?", default="SPY")
    bt.add_argument("--start", default="2016-01-01", help="first day of data to download")
    bt.add_argument("--demo", action="store_true", help="use made-up prices instead of Alpaca")
    bt.add_argument("--refresh", action="store_true", help="download fresh data instead of using the saved copy")
    bt.add_argument("--threshold", type=float, default=0.5)
    bt.add_argument("--cost", type=float, default=5.0, help="cost per trade in basis points (default 5)")

    sig = sub.add_parser("signal", help="today's prediction and paper trade")
    sig.add_argument("symbol", nargs="?", default="SPY")
    sig.add_argument("--model", default="logistic", choices=model.MODELS + model.BASELINES)
    sig.add_argument("--qty", type=float, default=1, help="shares to buy when the signal says buy")
    sig.add_argument("--threshold", type=float, default=0.5)
    sig.add_argument("--submit", action="store_true", help="send the order to the paper account")

    args = parser.parse_args(argv)
    if args.command == "backtest":
        run_backtest(args)
    else:
        run_signal(args)


def run_backtest(args) -> None:
    symbol = "DEMO" if args.demo else args.symbol.upper()
    bars = synthetic_bars() if args.demo else load_bars(symbol, args.start, refresh=args.refresh)
    df = build(bars)
    print(f"{symbol}: {len(df)} trading days, {df.index[0]:%d %b %Y} to {df.index[-1]:%d %b %Y}")

    results = []
    # "always up" is left out because it's the same as buy and hold
    for name in ["momentum"] + model.MODELS:
        prob = model.walk_forward(df, name)
        results.append(backtest.run(df, prob, args.threshold, args.cost))
    results.append(backtest.buy_and_hold(df, prob.index))

    tested = results[-1].returns.index
    print(f"Tested on {len(tested)} days from {tested[0]:%d %b %Y} (earlier days are only used for training)\n")
    print(backtest.table(results).to_string())
    chart = report.save_chart(symbol, results, " (random prices)" if args.demo else "")
    tbl = report.save_table(symbol, results)
    print(f"\nSaved {chart.name} and {tbl.name} in reports/")


def run_signal(args) -> None:
    symbol = args.symbol.upper()
    df = build(load_bars(symbol, refresh=True))
    prob = model.fit_latest(df, args.model)
    print(f"Latest bar: {df.index[-1]:%a %d %b %Y}, close {df['close'].iloc[-1]:.2f}")

    client = trade.paper_client()
    holding = trade.current_holding(client, symbol)
    p = trade.plan(symbol, prob, holding, args.qty, args.threshold)
    print(p.describe())
    if args.submit:
        print(trade.submit(client, p))
    elif p.action != "hold":
        print("Dry run. Add --submit to send this to your paper account.")


if __name__ == "__main__":
    main()
