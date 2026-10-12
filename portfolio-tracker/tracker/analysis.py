"""
One entry point that both the dashboard and the CLI report use.

`analyze` is pure (prices are passed in), so it can be tested offline.
`load_and_analyze` adds the Yahoo Finance calls on top.
"""

from dataclasses import dataclass

import pandas as pd

from . import data, metrics, portfolio


@dataclass
class Result:
    config: data.Config
    trades: pd.DataFrame          # journal with filled prices and cash flows
    history: pd.DataFrame         # daily position values, cash, total value
    benchmark: pd.Series          # benchmark closes on the same days
    holdings: pd.DataFrame        # current positions at live prices
    cash: float
    summary: dict
    sectors: pd.Series            # sector weights incl. cash


def analyze(config, trades, prices, live_prices, sectors):
    trades = portfolio.prepare(trades, prices)
    history = portfolio.build_history(trades, prices, config.starting_cash, config.start_date)
    bench = prices[config.benchmark].reindex(history.index).ffill()

    # Mark today's value at live prices, falling back to the latest close.
    tickers = sorted(trades["ticker"].unique())
    last_close = prices[tickers].ffill().iloc[-1]
    live = live_prices.reindex(tickers).fillna(last_close)
    cash = float(history["cash"].iloc[-1])
    holdings = portfolio.holdings_table(trades, live, sectors, cash)

    return Result(
        config=config,
        trades=trades,
        history=history,
        benchmark=bench,
        holdings=holdings,
        cash=cash,
        summary=metrics.summary(history["value"], bench, config.risk_free_rate),
        sectors=metrics.sector_allocation(holdings, cash),
    )


def load_and_analyze(config_path, trades_path):
    config = data.load_config(config_path)
    trades = data.load_trades(trades_path)
    tickers = sorted(trades["ticker"].unique())
    first = min(config.start_date, trades["date"].min()) - pd.Timedelta(days=7)
    prices = data.fetch_prices(tickers + [config.benchmark], start=first)
    live = data.fetch_live_prices(tickers)
    sectors = data.resolve_sectors(tickers, config.sectors)
    return analyze(config, trades, prices, live, sectors)
