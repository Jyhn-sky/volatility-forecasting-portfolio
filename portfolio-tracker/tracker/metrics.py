"""
Performance and risk metrics, computed from daily values.

All annualized figures assume 252 trading days a year.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(series):
    return series.pct_change().dropna()


def total_return(series):
    return series.iloc[-1] / series.iloc[0] - 1


def beta(port_ret, bench_ret):
    """Slope of portfolio returns regressed on benchmark returns."""
    p, b = port_ret.align(bench_ret, join="inner")
    if len(p) < 2 or b.var() == 0:
        return np.nan
    return np.cov(p, b, ddof=1)[0, 1] / b.var(ddof=1)


def jensen_alpha(port_ret, bench_ret, risk_free_rate):
    """Annualized return the portfolio earned beyond what its beta explains."""
    p, b = port_ret.align(bench_ret, join="inner")
    rf = risk_free_rate / TRADING_DAYS
    return ((p.mean() - rf) - beta(p, b) * (b.mean() - rf)) * TRADING_DAYS


def annualized_volatility(ret):
    return ret.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(ret, risk_free_rate):
    vol = annualized_volatility(ret)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (ret.mean() * TRADING_DAYS - risk_free_rate) / vol


def drawdown(series):
    """How far below its running peak the series is each day (0 = at a new high)."""
    return series / series.cummax() - 1


def max_drawdown(series):
    return drawdown(series).min()


def growth_of_100(series):
    return series / series.iloc[0] * 100


def sector_allocation(holdings, cash):
    """Share of total portfolio value in each sector, with cash as its own row."""
    by_sector = holdings.groupby("sector")["market_value"].sum()
    by_sector["Cash"] = cash
    return (by_sector / by_sector.sum()).sort_values(ascending=False)


def summary(port_value, bench_close, risk_free_rate):
    """Headline metrics for the dashboard and the CLI report."""
    bench_close = bench_close.reindex(port_value.index).ffill()
    pr, br = daily_returns(port_value), daily_returns(bench_close)
    port_tr, bench_tr = total_return(port_value), total_return(bench_close)
    return {
        "Portfolio value": port_value.iloc[-1],
        "Total return": port_tr,
        "Benchmark return": bench_tr,
        "Excess return": port_tr - bench_tr,
        "Beta": beta(pr, br),
        "Alpha (annual)": jensen_alpha(pr, br, risk_free_rate),
        "Correlation": pr.corr(br),
        "Volatility": annualized_volatility(pr),
        "Benchmark volatility": annualized_volatility(br),
        "Sharpe ratio": sharpe_ratio(pr, risk_free_rate),
        "Max drawdown": max_drawdown(port_value),
        "Benchmark max drawdown": max_drawdown(bench_close),
        "Current drawdown": drawdown(port_value).iloc[-1],
        "Trading days": len(port_value),
    }
