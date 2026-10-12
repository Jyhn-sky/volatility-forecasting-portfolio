"""
Turning the trade journal into positions, a daily value history and a
current holdings table.

Trades are assumed to fill at the day's close, so buying or selling moves
money between cash and stock without creating a fake jump in returns.
"""

import numpy as np
import pandas as pd


def fill_trade_prices(trades, prices):
    """Fill blank trade prices with the close on (or just before) the trade date."""
    trades = trades.copy()
    for i, row in trades[trades["price"].isna()].iterrows():
        if row["ticker"] not in prices.columns:
            raise ValueError(f"No price history for {row['ticker']}")
        series = prices[row["ticker"]].dropna()
        close = series.asof(row["date"])
        if pd.isna(close):
            raise ValueError(
                f"No close for {row['ticker']} on or before {row['date'].date()}; "
                "enter the price manually in trades.csv")
        trades.at[i, "price"] = close
    return trades


def add_cash_flows(trades):
    """Signed share change and the cash it costs (negative) or brings in (positive)."""
    trades = trades.copy()
    sign = np.where(trades["action"] == "SELL", -1, 1)
    trades["signed_shares"] = sign * trades["shares"]
    trades["cash_flow"] = -trades["signed_shares"] * trades["price"] - trades["fees"]
    return trades


def check_no_short_positions(trades):
    running = trades.groupby("ticker")["signed_shares"].cumsum()
    oversold = trades[running < -1e-9]
    if not oversold.empty:
        r = oversold.iloc[0]
        raise ValueError(f"Selling more {r['ticker']} than you own on {r['date'].date()}")


def build_history(trades, prices, starting_cash, start_date):
    """
    Daily portfolio history from start_date onward.

    Returns a DataFrame with one '<TICKER>' column of position value per
    holding, plus 'equities', 'cash' and 'value'.
    """
    prices = prices.loc[prices.index >= start_date].ffill()
    if prices.empty:
        raise ValueError("No prices on or after the start date")
    tickers = sorted(trades["ticker"].unique())

    # Shares and cash as of each day: cumulative sum of trades dated on or before it.
    # Trades on non-trading days roll into the next trading day.
    by_day = trades.groupby(["date", "ticker"])["signed_shares"].sum().unstack(fill_value=0)
    all_days = prices.index.union(by_day.index)
    shares = (by_day.reindex(all_days, fill_value=0).cumsum()
              .reindex(prices.index, method="ffill").fillna(0)
              .reindex(columns=tickers, fill_value=0))
    flows = trades.groupby("date")["cash_flow"].sum()
    cash = (starting_cash + flows.reindex(prices.index.union(flows.index), fill_value=0)
            .cumsum().reindex(prices.index, method="ffill"))

    values = shares * prices[tickers]
    history = values.copy()
    history["equities"] = values.sum(axis=1)
    history["cash"] = cash
    history["value"] = history["equities"] + history["cash"]
    return history


def holdings_table(trades, live_prices, sectors, cash):
    """Current positions with cost, live value, weight and unrealized P/L."""
    rows = []
    for ticker, t in trades.groupby("ticker"):
        shares = t["signed_shares"].sum()
        if abs(shares) < 1e-9:
            continue
        buys = t[t["action"] == "BUY"]
        avg_cost = (buys["shares"] * buys["price"]).sum() / buys["shares"].sum()
        price = live_prices.get(ticker, np.nan)
        rows.append({
            "ticker": ticker,
            "sector": sectors.get(ticker, "Unknown"),
            "shares": shares,
            "avg_cost": avg_cost,
            "price": price,
            "market_value": shares * price,
            "cost_basis": shares * avg_cost,
        })
    table = pd.DataFrame(rows, columns=["ticker", "sector", "shares", "avg_cost",
                                        "price", "market_value", "cost_basis"])
    total = table["market_value"].sum() + cash
    table["unrealized_pl"] = table["market_value"] - table["cost_basis"]
    table["return"] = table["unrealized_pl"] / table["cost_basis"]
    table["weight"] = table["market_value"] / total
    return table.sort_values("market_value", ascending=False).reset_index(drop=True)


def prepare(trades, prices):
    """Fill prices, compute cash flows and sanity-check the journal."""
    trades = add_cash_flows(fill_trade_prices(trades, prices))
    check_no_short_positions(trades)
    return trades
