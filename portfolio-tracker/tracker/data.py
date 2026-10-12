"""
Loading the config and trade journal, and pulling prices from Yahoo Finance.

Everything that touches the network lives here, so the rest of the package
can be tested with made-up prices.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TRADE_COLUMNS = ["date", "ticker", "action", "shares", "price", "fees",
                 "thesis", "catalyst", "key_risk", "exit_trigger"]


@dataclass
class Config:
    name: str
    start_date: pd.Timestamp
    starting_cash: float
    benchmark: str
    risk_free_rate: float
    sectors: dict = field(default_factory=dict)


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    p = raw["portfolio"]
    return Config(
        name=p.get("name", "Portfolio"),
        start_date=pd.Timestamp(p["start_date"]),
        starting_cash=float(p["starting_cash"]),
        benchmark=p.get("benchmark", "SPY"),
        risk_free_rate=float(p.get("risk_free_rate", 0.0)),
        sectors={k.upper(): v for k, v in raw.get("sectors", {}).items()},
    )


def load_trades(path="data/trades.csv"):
    """Read and validate the trade journal. A blank price means 'filled at that day's close'."""
    trades = pd.read_csv(path, dtype={"thesis": str, "catalyst": str,
                                      "key_risk": str, "exit_trigger": str})
    missing = set(TRADE_COLUMNS) - set(trades.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    trades["date"] = pd.to_datetime(trades["date"])
    trades["ticker"] = trades["ticker"].str.strip().str.upper()
    trades["action"] = trades["action"].str.strip().str.upper()
    trades["fees"] = trades["fees"].fillna(0.0).astype(float)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    for col in ["thesis", "catalyst", "key_risk", "exit_trigger"]:
        trades[col] = trades[col].fillna("")

    bad_action = ~trades["action"].isin(["BUY", "SELL"])
    if bad_action.any():
        rows = (trades.index[bad_action] + 2).tolist()   # +2: header row, 1-based
        raise ValueError(f"action must be BUY or SELL (check rows {rows} of {path})")
    if (trades["shares"] <= 0).any():
        raise ValueError("shares must be positive; use action SELL to reduce a position")

    return trades.sort_values("date", kind="stable").reset_index(drop=True)


def fetch_prices(tickers, start, end=None):
    """Daily closing prices, one column per ticker."""
    import yfinance as yf

    tickers = sorted(set(tickers))
    data = yf.download(tickers, start=start, end=end, auto_adjust=False,
                       progress=False, group_by="column")
    if data.empty:
        raise RuntimeError("No price data returned. Check tickers and your internet connection.")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise RuntimeError(f"No prices found for: {missing}")
    return close[tickers]


def fetch_live_prices(tickers):
    """Latest trade price for each ticker (Yahoo quotes can be ~15 min delayed)."""
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            out[t] = float(yf.Ticker(t).fast_info["last_price"])
        except Exception:
            out[t] = float("nan")
    return pd.Series(out)


def resolve_sectors(tickers, known):
    """Use the config's sectors, falling back to Yahoo Finance for anything missing."""
    sectors = {}
    for t in tickers:
        if t in known:
            sectors[t] = known[t]
            continue
        try:
            import yfinance as yf
            sectors[t] = yf.Ticker(t).info.get("sector") or "Unknown"
        except Exception:
            sectors[t] = "Unknown"
    return sectors


def project_root():
    return Path(__file__).resolve().parent.parent
