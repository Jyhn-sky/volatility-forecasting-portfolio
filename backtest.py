"""
backtest.py

Turns per-asset volatility forecasts into portfolio weights and backtests
the result. Two allocation schemes:
  - "risk_parity": weight inversely proportional to forecast vol
    (equal risk contribution, ignoring correlation -- simple and robust)
  - "min_variance": full mean-variance minimum-variance solution using a
    shrinkage covariance estimate, scaled by the forecast vols

Includes transaction costs and a realistic 1-day rebalancing lag (you
trade on forecasts made using only data available *before* today).
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def risk_parity_weights(vol_forecasts: pd.Series):
    """Weight inversely proportional to forecast volatility, normalized to sum to 1."""
    inv_vol = 1.0 / vol_forecasts.clip(lower=1e-4)
    return inv_vol / inv_vol.sum()


def min_variance_weights(vol_forecasts: pd.Series, corr: pd.DataFrame, shrinkage=0.2):
    """
    Minimum-variance weights using forecast vols + a shrunk correlation
    matrix (shrunk toward the identity to avoid unstable/extreme weights
    from noisy correlation estimates -- a standard practical fix).
    """
    tickers = vol_forecasts.index
    corr = corr.loc[tickers, tickers]
    shrunk_corr = (1 - shrinkage) * corr + shrinkage * np.eye(len(tickers))
    vols = vol_forecasts.values.reshape(-1, 1)
    cov = shrunk_corr.values * (vols @ vols.T)

    ones = np.ones((len(tickers), 1))
    try:
        inv_cov = np.linalg.pinv(cov)
        w = inv_cov @ ones
        w = w / w.sum()
        w = np.clip(w.flatten(), 0, None)  # long-only
        w = w / w.sum()
    except np.linalg.LinAlgError:
        w = np.ones(len(tickers)) / len(tickers)

    return pd.Series(w, index=tickers)


def backtest_portfolio(returns: pd.DataFrame, vol_forecast_panel: pd.DataFrame,
                        forecast_col="ml_pred", scheme="risk_parity",
                        rebalance_freq=21, transaction_cost_bps=5, lag_days=1):
    """
    returns: wide DataFrame of daily log returns, columns=tickers
    vol_forecast_panel: long DataFrame with ['ticker', forecast_col], indexed by date
    scheme: 'risk_parity' or 'min_variance'
    lag_days: forecasts made on date T are only tradable from T+lag_days onward
    """
    forecast_wide = vol_forecast_panel.pivot_table(
        index=vol_forecast_panel.index, columns="ticker", values=forecast_col
    )
    common_tickers = [c for c in returns.columns if c in forecast_wide.columns]
    returns = returns[common_tickers]
    forecast_wide = forecast_wide[common_tickers].shift(lag_days)  # apply trading lag

    all_dates = returns.index.intersection(forecast_wide.dropna().index)
    all_dates = np.sort(all_dates)
    if len(all_dates) == 0:
        raise ValueError("No overlapping dates between returns and forecasts after lag.")

    rebalance_dates = set(all_dates[::rebalance_freq])

    weights_history = []
    port_returns = []
    prev_weights = pd.Series(1.0 / len(common_tickers), index=common_tickers)
    corr_lookback = 126

    for date in all_dates:
        if date in rebalance_dates:
            vol_fc = forecast_wide.loc[date].dropna()
            if scheme == "risk_parity":
                new_weights = risk_parity_weights(vol_fc)
            else:
                hist = returns.loc[:date].iloc[-corr_lookback:]
                corr = hist[vol_fc.index].corr()
                new_weights = min_variance_weights(vol_fc, corr)
            new_weights = new_weights.reindex(common_tickers).fillna(0)

            turnover = (new_weights - prev_weights).abs().sum()
            cost = turnover * (transaction_cost_bps / 1e4)
            prev_weights = new_weights
        else:
            cost = 0.0

        weights_history.append(prev_weights.copy())
        day_return = (prev_weights * returns.loc[date]).sum() - cost
        port_returns.append(day_return)

    port_returns = pd.Series(port_returns, index=all_dates, name="portfolio_return")
    weights_df = pd.DataFrame(weights_history, index=all_dates)
    return port_returns, weights_df


def performance_summary(port_returns: pd.Series):
    ann_return = port_returns.mean() * TRADING_DAYS
    ann_vol = port_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + port_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()

    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


if __name__ == "__main__":
    from data_pipeline import generate_synthetic_data, compute_features

    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PG"]
    prices, vix = generate_synthetic_data(tickers, n_days=800)
    panel = compute_features(prices, vix)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # use trailing realized vol as a stand-in "forecast" for this quick smoke test
    fake_forecast = panel[["ticker", "rvol_21d"]].rename(columns={"rvol_21d": "ml_pred"})
    port_returns, weights = backtest_portfolio(log_returns, fake_forecast, scheme="risk_parity")

    print(performance_summary(port_returns))

    equal_weight_returns = log_returns[weights.columns].mean(axis=1).reindex(port_returns.index)
    print("Equal-weight baseline:", performance_summary(equal_weight_returns))
