"""Offline tests: made-up prices, no network."""

import numpy as np
import pandas as pd
import pytest

from tracker import metrics, portfolio
from tracker.analysis import analyze
from tracker.data import Config

DAYS = pd.bdate_range("2026-01-02", periods=60)


def make_prices(seed=0):
    rng = np.random.default_rng(seed)
    bench = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(DAYS)))
    aaa = 50 * np.cumprod(1 + rng.normal(0.001, 0.02, len(DAYS)))
    bbb = 20 * np.cumprod(1 + rng.normal(0.0, 0.015, len(DAYS)))
    return pd.DataFrame({"BENCH": bench, "AAA": aaa, "BBB": bbb}, index=DAYS)


def make_trades(rows):
    df = pd.DataFrame(rows, columns=["date", "ticker", "action", "shares", "price", "fees"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ["thesis", "catalyst", "key_risk", "exit_trigger"]:
        df[col] = ""
    return df


def config(**kw):
    base = dict(name="Test", start_date=DAYS[0], starting_cash=10_000.0,
                benchmark="BENCH", risk_free_rate=0.0)
    return Config(**{**base, **kw})


def test_blank_price_fills_with_close():
    prices = make_prices()
    trades = make_trades([(DAYS[3], "AAA", "BUY", 10, np.nan, 0)])
    filled = portfolio.fill_trade_prices(trades, prices)
    assert filled.loc[0, "price"] == pytest.approx(prices.loc[DAYS[3], "AAA"])


def test_weekend_trade_uses_previous_close():
    prices = make_prices()
    saturday = DAYS[5] + pd.Timedelta(days=1)   # DAYS[5] is a Friday
    trades = make_trades([(saturday, "AAA", "BUY", 10, np.nan, 0)])
    filled = portfolio.fill_trade_prices(trades, prices)
    assert filled.loc[0, "price"] == pytest.approx(prices.loc[DAYS[5], "AAA"])


def test_buying_at_close_does_not_change_value():
    prices = make_prices()
    trades = portfolio.prepare(make_trades([(DAYS[10], "AAA", "BUY", 50, np.nan, 0)]), prices)
    hist = portfolio.build_history(trades, prices, 10_000.0, DAYS[0])
    # all cash before the trade, and value is unchanged on the trade day itself
    assert (hist.loc[:DAYS[9], "value"] == 10_000.0).all()
    assert hist.loc[DAYS[10], "value"] == pytest.approx(10_000.0)
    # afterwards it moves with the stock
    expected = 10_000 + 50 * (prices.loc[DAYS[-1], "AAA"] - prices.loc[DAYS[10], "AAA"])
    assert hist["value"].iloc[-1] == pytest.approx(expected)


def test_sell_and_fees_flow_into_cash():
    prices = make_prices()
    trades = portfolio.prepare(make_trades([
        (DAYS[0], "AAA", "BUY", 100, 50.0, 5),
        (DAYS[5], "AAA", "SELL", 40, 60.0, 5),
    ]), prices)
    hist = portfolio.build_history(trades, prices, 10_000.0, DAYS[0])
    assert hist["cash"].iloc[-1] == pytest.approx(10_000 - 5000 - 5 + 2400 - 5)
    assert hist["AAA"].iloc[-1] == pytest.approx(60 * prices["AAA"].iloc[-1])


def test_overselling_is_rejected():
    prices = make_prices()
    trades = make_trades([
        (DAYS[0], "AAA", "BUY", 10, np.nan, 0),
        (DAYS[5], "AAA", "SELL", 11, np.nan, 0),
    ])
    with pytest.raises(ValueError, match="more AAA than you own"):
        portfolio.prepare(trades, prices)


def test_holding_only_the_benchmark_gives_beta_one_alpha_zero():
    prices = make_prices()
    prices["CLONE"] = prices["BENCH"]
    trades = make_trades([(DAYS[0], "CLONE", "BUY", 100, np.nan, 0)])
    start_cash = 100 * prices["BENCH"].iloc[0]          # fully invested, no cash drag
    r = analyze(config(starting_cash=start_cash), trades, prices,
                pd.Series(dtype=float), {"CLONE": "Index"})
    assert r.summary["Beta"] == pytest.approx(1.0)
    assert r.summary["Alpha (annual)"] == pytest.approx(0.0, abs=1e-12)
    assert r.summary["Excess return"] == pytest.approx(0.0, abs=1e-12)
    assert r.summary["Correlation"] == pytest.approx(1.0)


def test_drawdown():
    s = pd.Series([100, 120, 90, 110, 130])
    assert metrics.max_drawdown(s) == pytest.approx(90 / 120 - 1)
    assert metrics.drawdown(s).iloc[-1] == 0


def test_sector_allocation_sums_to_one_with_cash():
    prices = make_prices()
    trades = make_trades([
        (DAYS[0], "AAA", "BUY", 50, np.nan, 0),
        (DAYS[0], "BBB", "BUY", 100, np.nan, 0),
    ])
    r = analyze(config(), trades, prices, pd.Series({"AAA": 55.0, "BBB": 21.0}),
                {"AAA": "Tech", "BBB": "Energy"})
    assert r.sectors.sum() == pytest.approx(1.0)
    assert set(r.sectors.index) == {"Tech", "Energy", "Cash"}
    assert r.holdings.set_index("ticker").loc["AAA", "price"] == 55.0
    assert r.holdings["weight"].sum() + r.sectors["Cash"] == pytest.approx(1.0)
