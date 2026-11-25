"""Chart and table for a backtest."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, just save the file
import matplotlib.pyplot as plt  # noqa: E402

from .backtest import Result, table  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def save_chart(symbol: str, results: list[Result], title_note: str = "") -> Path:
    REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    for r in results:
        equity = (1 + r.returns).cumprod()
        style = {"color": "black", "linewidth": 2.2} if r.name == "buy_and_hold" else {"linewidth": 1.3}
        ax.plot(equity.index, equity, label=r.name.replace("_", " "), **style)
    ax.set_title(f"{symbol}: $1 invested, walk-forward predictions after costs{title_note}")
    ax.set_ylabel("Value of $1")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = REPORTS / f"{symbol.lower()}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_table(symbol: str, results: list[Result]) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{symbol.lower()}.md"
    path.write_text(table(results).to_markdown() + "\n")
    return path
