"""
Prints the portfolio summary and holdings in the terminal.

Usage:
    python report.py               # print the report
    python report.py --save        # also write CSVs to output/
"""

import argparse
from pathlib import Path

import pandas as pd

from tracker.analysis import load_and_analyze

ROOT = Path(__file__).resolve().parent
PCT_KEYS = {"Total return", "Benchmark return", "Excess return", "Alpha (annual)",
            "Max drawdown", "Benchmark max drawdown", "Current drawdown"}
VOL_KEYS = {"Volatility", "Benchmark volatility"}


def fmt(key, value):
    if pd.isna(value):
        return "n/a"
    if key in PCT_KEYS:
        return f"{value:+.2%}"
    if key in VOL_KEYS:
        return f"{value:.2%}"
    if key == "Portfolio value":
        return f"${value:,.2f}"
    if key == "Trading days":
        return f"{int(value)}"
    return f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "config.toml")
    parser.add_argument("--trades", default=ROOT / "data" / "trades.csv")
    parser.add_argument("--save", action="store_true", help="write CSVs to output/")
    args = parser.parse_args()

    r = load_and_analyze(args.config, args.trades)
    c = r.config
    print(f"\n{c.name}  |  benchmark {c.benchmark}  |  since {c.start_date.date()}\n")
    for key, value in r.summary.items():
        print(f"  {key:<24}{fmt(key, value):>14}")

    print("\nHoldings")
    h = r.holdings.copy()
    print(h.to_string(index=False, formatters={
        "shares": "{:,.0f}".format, "avg_cost": "${:,.2f}".format,
        "price": "${:,.2f}".format, "market_value": "${:,.0f}".format,
        "cost_basis": "${:,.0f}".format, "unrealized_pl": "${:+,.0f}".format,
        "return": "{:+.1%}".format, "weight": "{:.1%}".format}))
    print(f"\n  Cash: ${r.cash:,.2f}")

    print("\nSector allocation")
    for sector, w in r.sectors.items():
        print(f"  {sector:<26}{w:>7.1%}")

    if args.save:
        out = ROOT / "output"
        out.mkdir(exist_ok=True)
        r.history.assign(benchmark=r.benchmark).to_csv(out / "history.csv")
        r.holdings.to_csv(out / "holdings.csv", index=False)
        pd.Series(r.summary).to_csv(out / "summary.csv", header=["value"])
        print(f"\nSaved CSVs to {out}")


if __name__ == "__main__":
    main()
