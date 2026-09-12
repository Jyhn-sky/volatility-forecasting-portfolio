"""
data_pipeline.py

Fetches daily price data for a universe of tickers + VIX, and builds the
feature set used by both the GARCH baseline and the ML volatility model.

Real data: uses yfinance (requires internet access when you run this locally).
Offline/demo mode: generates synthetic GBM-with-vol-clustering price paths so
you can test the full pipeline without network access. Toggle with `demo=True`.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def fetch_real_data(tickers, start, end, vix_ticker="^VIX"):
    """Pull daily adjusted close prices + VIX from Yahoo Finance."""
    import yfinance as yf

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    vix = yf.download(vix_ticker, start=start, end=end, auto_adjust=True)["Close"]
    vix = vix.rename("VIX") if isinstance(vix, pd.Series) else vix.iloc[:, 0].rename("VIX")

    return prices.dropna(how="all"), vix.dropna()


def generate_synthetic_data(tickers, n_days=1500, seed=42):
    """
    Generate synthetic price paths with realistic stylized facts:
    volatility clustering (via a simple GARCH(1,1) DGP) and fat tails.
    Used only for offline testing of the pipeline logic.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    # GARCH(1,1) parameters for the *true* data-generating process
    omega, alpha, beta = 1e-6, 0.08, 0.88
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    n_days = len(dates)  # bdate_range can return periods-1 depending on the anchor date

    price_data = {}
    all_returns = []
    for i, ticker in enumerate(tickers):
        h = np.zeros(n_days)          # conditional variance
        eps = np.zeros(n_days)        # innovations
        h[0] = omega / (1 - alpha - beta)
        # Student-t innovations for fat tails
        z = rng.standard_t(df=5, size=n_days) / np.sqrt(5 / 3)
        for t in range(1, n_days):
            h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
            eps[t] = np.sqrt(h[t]) * z[t]

        drift = 0.0002 + 0.00005 * i  # tiny per-asset drift variation
        log_returns = drift + eps
        prices = 100 * np.exp(np.cumsum(log_returns))
        price_data[ticker] = prices
        all_returns.append(log_returns)

    prices_df = pd.DataFrame(price_data, index=dates)

    # Synthetic VIX-like series: annualized vol of an equal-weight basket,
    # smoothed, roughly mean-reverting around 16-20.
    basket_returns = np.mean(all_returns, axis=0)
    realized = pd.Series(basket_returns, index=dates).rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
    vix = (realized.ffill().bfill() * 1.15 + rng.normal(0, 1.0, n_days)).clip(lower=9)
    vix.name = "VIX"

    return prices_df, vix


def compute_features(prices: pd.DataFrame, vix: pd.Series, vol_horizon=21):
    """
    Build a per-ticker, per-day feature panel.

    Target: forward-looking realized volatility over `vol_horizon` days
            (annualized), i.e. what we're trying to forecast.
    Features: trailing realized vol at multiple windows, trailing returns,
              volume proxies (skipped here for simplicity; add if you pull
              Volume from yfinance), and the VIX level/change as an
              exogenous market-wide risk signal.
    """
    log_returns = np.log(prices / prices.shift(1))

    panels = []
    for ticker in prices.columns:
        r = log_returns[ticker].dropna()

        df = pd.DataFrame(index=r.index)
        df["ticker"] = ticker
        df["log_return"] = r

        # trailing realized vol features (annualized)
        for w in (5, 10, 21, 63):
            df[f"rvol_{w}d"] = r.rolling(w).std() * np.sqrt(TRADING_DAYS)

        # momentum / mean-reversion style features
        df["mom_21d"] = r.rolling(21).sum()
        df["mom_63d"] = r.rolling(63).sum()

        # exogenous market vol signal
        df["vix"] = vix.reindex(df.index).ffill()
        df["vix_chg_5d"] = df["vix"].diff(5)

        # TARGET 1: forward realized vol over the next `vol_horizon` days
        # (used for the point-forecast model, evaluated against GARCH/EWMA)
        fwd_var = (r.shift(-1).rolling(vol_horizon).var().shift(-(vol_horizon - 1)))
        df["target_fwd_rvol"] = np.sqrt(fwd_var) * np.sqrt(TRADING_DAYS)

        # TARGET 2: next single-day log return (used for the VaR quantile
        # model -- VaR is a statement about the return distribution, not
        # about forward volatility, so it needs its own target)
        df["target_next_return"] = r.shift(-1)

        panels.append(df)

    panel = pd.concat(panels).dropna()
    panel.index.name = None  # normalize: yfinance labels its date index 'Date',
    # synthetic data leaves it unnamed -- force both to the same (unnamed) state
    # so every downstream reset_index() call produces a column named 'index'
    # regardless of data source.
    return panel


def walk_forward_splits(panel: pd.DataFrame, n_splits=5, min_train_frac=0.5):
    """
    Yield (train_idx, test_idx) index arrays for expanding-window
    walk-forward validation, split on the date axis (not asset axis) so
    there is no look-ahead leakage across the split boundary.
    """
    dates = np.sort(panel.index.unique())
    n = len(dates)
    min_train = int(n * min_train_frac)
    fold_size = (n - min_train) // n_splits

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_end = min_train + (i + 1) * fold_size if i < n_splits - 1 else n
        train_dates = dates[:train_end]
        test_dates = dates[train_end:test_end]
        if len(test_dates) == 0:
            continue
        yield panel[panel.index.isin(train_dates)], panel[panel.index.isin(test_dates)]


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PG"]
    prices, vix = generate_synthetic_data(tickers)
    panel = compute_features(prices, vix)
    print(panel.head())
    print(f"\nPanel shape: {panel.shape}, date range: {panel.index.min()} to {panel.index.max()}")
