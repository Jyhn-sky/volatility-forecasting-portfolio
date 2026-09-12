"""
evaluate.py

Forecast evaluation. Two things matter here that a lot of finance-ML
projects skip:

1. QLIKE loss -- the standard metric in the volatility forecasting
   literature (Patton, 2011). It penalizes underforecasting vol more
   than MSE does, which matters a lot for risk management (you'd much
   rather over- than under-estimate risk).
2. VaR calibration -- if you claim a 5% VaR / quantile forecast, the
   actual breach rate on held-out data should be close to 5%. A model
   that "predicts" tail risk but is miscalibrated is worse than useless
   in a risk context because it gives false confidence.
"""

import numpy as np
import pandas as pd


def qlike_loss(y_true, y_pred):
    """
    QLIKE loss on variances (lower is better). Expects vol (not variance)
    inputs and squares internally. Undefined/unstable near zero, so we
    floor predictions.
    """
    var_true = np.asarray(y_true) ** 2
    var_pred = np.clip(np.asarray(y_pred) ** 2, 1e-8, None)
    return np.mean(var_true / var_pred - np.log(var_true / var_pred) - 1)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def compare_forecasts(merged: pd.DataFrame, target_col="target_fwd_rvol",
                       forecast_cols=("ewma_fwd_rvol", "garch_fwd_rvol", "ml_pred")):
    """
    merged: DataFrame containing the target column and one or more
    forecast columns, already aligned/joined on (ticker, date).
    Returns a small leaderboard DataFrame.
    """
    rows = []
    for col in forecast_cols:
        d = merged[[target_col, col]].dropna()
        if len(d) == 0:
            continue
        rows.append({
            "model": col,
            "n_obs": len(d),
            "rmse": rmse(d[target_col], d[col]),
            "qlike": qlike_loss(d[target_col], d[col]),
        })
    return pd.DataFrame(rows).sort_values("qlike")


def var_calibration(returns: pd.Series, var_forecast: pd.Series, quantile=0.05):
    """
    Checks how often realized returns breach the forecast VaR quantile.
    A well-calibrated 5% VaR should be breached ~5% of the time -- not
    2%, not 15%. This is the single most important sanity check for any
    tail-risk model before it's trusted for risk management.
    """
    aligned = pd.DataFrame({"ret": returns, "var": -var_forecast.abs()}).dropna()
    breaches = (aligned["ret"] < aligned["var"]).mean()
    return {
        "target_breach_rate": quantile,
        "actual_breach_rate": breaches,
        "n_obs": len(aligned),
        "well_calibrated": abs(breaches - quantile) < 0.02,
    }


if __name__ == "__main__":
    # quick smoke test with synthetic numbers
    rng = np.random.default_rng(0)
    y_true = np.abs(rng.normal(0.15, 0.03, 500))
    y_pred_good = y_true + rng.normal(0, 0.01, 500)
    y_pred_bad = np.full(500, 0.15)

    merged = pd.DataFrame({
        "target_fwd_rvol": y_true, "good_model": y_pred_good, "bad_model": y_pred_bad
    })
    print(compare_forecasts(merged, forecast_cols=("good_model", "bad_model")))
