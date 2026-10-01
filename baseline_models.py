"""
baseline_models.py

Classical econometric volatility baselines: EWMA (RiskMetrics-style) and
GARCH(1,1). These exist so the ML model has something honest to beat -- if
your LSTM can't outperform a GARCH(1,1), that's an important (and common!)
finding, not a failure of the project.
"""

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252


def ewma_forecast(returns: pd.Series, lam=0.94, horizon=21):
    """
    RiskMetrics-style EWMA volatility forecast. Returns the *current*
    conditional vol estimate (constant forecast over the horizon, since
    EWMA has no mean-reversion built in).
    """
    var = returns.iloc[0] ** 2
    for r in returns.iloc[1:]:
        var = lam * var + (1 - lam) * r ** 2
    daily_vol = np.sqrt(var)
    return daily_vol * np.sqrt(TRADING_DAYS)


def fit_garch(returns: pd.Series):
    """
    Fit a GARCH(1,1) with Student-t innovations (fatter tails than
    Gaussian, usually a better fit for daily equity returns).
    Returns are rescaled to percentage points as `arch` expects that
    scale for numerical stability.
    """
    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off")
    return res


def garch_filter(params, returns: pd.Series):
    """
    Run an already-fitted GARCH(1,1) over `returns` without re-estimating
    its parameters. This updates the conditional variance with every new
    return, which is cheap, so the forecast can move daily even though the
    (slow) parameter fit only happens every few weeks.
    """
    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="t", mean="Zero")
    return am.fix(params)


def garch_forecast(fitted_res, horizon=21):
    """
    Forecast annualized volatility `horizon` days ahead from a fitted
    GARCH model, aggregating the daily variance path.
    """
    f = fitted_res.forecast(horizon=horizon, reindex=False)
    daily_var = f.variance.values[-1] / (100 ** 2)  # undo the *100 rescale
    avg_daily_var = daily_var.mean()
    return np.sqrt(avg_daily_var) * np.sqrt(TRADING_DAYS)


def rolling_baseline_forecasts(panel: pd.DataFrame, horizon=21, refit_every=21):
    """
    Produce EWMA and GARCH forward-vol forecasts per ticker. GARCH
    parameters are re-estimated every `refit_every` days (refitting daily
    is slow and barely changes them), but the conditional variance is
    updated with each new day's return, so the forecast still reacts
    daily. Without that daily update the GARCH forecast would sit frozen
    at the same value for `refit_every` days at a time.
    Returns a DataFrame aligned to panel's index with baseline columns
    added, so it can be merged directly for evaluation.
    """
    results = []
    for ticker, g in panel.groupby("ticker"):
        g = g.sort_index()
        returns = g["log_return"]
        dates = g.index

        ewma_preds = np.full(len(dates), np.nan)
        garch_preds = np.full(len(dates), np.nan)

        params = None
        min_hist = 252
        for i in range(min_hist, len(dates)):
            hist = returns.iloc[:i]
            ewma_preds[i] = ewma_forecast(hist.iloc[-500:], horizon=horizon)

            if (i - min_hist) % refit_every == 0 or params is None:
                try:
                    params = fit_garch(hist.iloc[-750:]).params
                except Exception:
                    params = None
            if params is not None:
                try:
                    filtered = garch_filter(params, hist.iloc[-750:])
                    garch_preds[i] = garch_forecast(filtered, horizon=horizon)
                except Exception:
                    garch_preds[i] = np.nan

        out = pd.DataFrame(
            {"ticker": ticker, "ewma_fwd_rvol": ewma_preds, "garch_fwd_rvol": garch_preds},
            index=dates,
        )
        results.append(out)

    return pd.concat(results)


if __name__ == "__main__":
    # Minimal self-test on a tiny hand-built price series (same idea as the
    # one in data_pipeline.py). Not part of the real run path.
    from data_pipeline import compute_features

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=400)
    prices = pd.DataFrame({"TEST": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(dates))))},
                          index=dates)
    vix = pd.Series(rng.normal(18, 3, len(dates)).clip(min=9), index=dates, name="VIX")
    panel = compute_features(prices, vix)

    baselines = rolling_baseline_forecasts(panel)
    print(baselines.dropna().head())
    print(f"\n{baselines.dropna().shape[0]} forecast rows produced")
