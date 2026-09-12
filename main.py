"""
main.py

End-to-end pipeline:
  1. Load data (real via yfinance, or synthetic for offline demo)
  2. Build features
  3. Walk-forward split
  4. Fit GARCH/EWMA baselines + train LSTM on the training window
  5. Generate forecasts on the held-out window
  6. Evaluate forecast accuracy (RMSE, QLIKE) and VaR calibration
  7. Backtest portfolios built from each forecast source
  8. Save comparison plots + a results table

Usage:
    python main.py --demo                     # synthetic data, fast, no internet needed
    python main.py --tickers AAPL MSFT JPM XOM PG --start 2018-01-01 --end 2024-01-01
"""

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline import fetch_real_data, generate_synthetic_data, compute_features
from baseline_models import rolling_baseline_forecasts
from ml_model import train_model, predict, FEATURE_COLS
from backtest import backtest_portfolio, performance_summary
from evaluate import compare_forecasts, var_calibration

warnings.filterwarnings("ignore")


def run(tickers, demo, start, end, train_frac=0.7, out_dir="."):
    print(f"\n=== Loading data ({'synthetic demo' if demo else 'real via yfinance'}) ===")
    if demo:
        prices, vix = generate_synthetic_data(tickers, n_days=1500)
    else:
        prices, vix = fetch_real_data(tickers, start, end)

    print(f"Price data: {prices.shape}, date range {prices.index.min()} to {prices.index.max()}")

    print("\n=== Building features ===")
    panel = compute_features(prices, vix)
    dates = np.sort(panel.index.unique())
    split_idx = int(len(dates) * train_frac)
    split_date = dates[split_idx]
    train_panel = panel[panel.index <= split_date]
    test_panel = panel[panel.index > split_date]
    print(f"Train: {train_panel.index.min()} -> {train_panel.index.max()} ({len(train_panel)} rows)")
    print(f"Test:  {test_panel.index.min()} -> {test_panel.index.max()} ({len(test_panel)} rows)")

    print("\n=== Fitting GARCH / EWMA baselines on test window ===")
    baseline_fc = rolling_baseline_forecasts(panel)  # walks its own internal history per date
    baseline_fc = baseline_fc[baseline_fc.index > split_date]

    print("\n=== Training LSTM (point forecast) ===")
    val_cut = train_panel.index.unique()[int(len(train_panel.index.unique()) * 0.85)]
    lstm_train = train_panel[train_panel.index <= val_cut]
    lstm_val = train_panel[train_panel.index > val_cut]
    model, scaler, history = train_model(lstm_train, lstm_val, mode="point", epochs=25, verbose=True)

    print("\n=== Training LSTM (5% quantile, for VaR) ===")
    q_model, q_scaler, _ = train_model(lstm_train, lstm_val, mode="quantile", quantile=0.05,
                                        epochs=25, verbose=False)

    print("\n=== Generating test-set predictions ===")
    ml_preds = predict(model, test_panel, scaler, target_col="target_fwd_rvol")
    q_preds = predict(q_model, test_panel, q_scaler, target_col="target_next_return").rename(
        columns={"ml_pred": "ml_var_pred"})

    # merge everything for evaluation: index by (ticker, date)
    target = test_panel.reset_index().rename(columns={"index": "date"})[["date", "ticker", "target_fwd_rvol", "log_return"]]
    merged = target.merge(ml_preds.reset_index(), on=["date", "ticker"], how="left")
    merged = merged.merge(baseline_fc.reset_index().rename(columns={"index": "date"}), on=["date", "ticker"], how="left")
    merged = merged.set_index("date")

    print("\n=== Forecast accuracy: ML vs GARCH vs EWMA ===")
    leaderboard = compare_forecasts(merged, forecast_cols=("ewma_fwd_rvol", "garch_fwd_rvol", "ml_pred"))
    print(leaderboard.to_string(index=False))

    print("\n=== VaR calibration (5% return-quantile model) ===")
    # Compare against target_next_return -- the *exact* quantity the quantile
    # model was trained to predict -- rather than a same-day return column,
    # which would be off by construction since the target is next-day return.
    for ticker, g in test_panel.groupby("ticker"):
        q_g = q_preds[q_preds["ticker"] == ticker]
        aligned = g[["target_next_return"]].join(q_g[["ml_var_pred"]], how="inner")
        if len(aligned) < 30:
            continue
        calib = var_calibration(aligned["target_next_return"], aligned["ml_var_pred"], quantile=0.05)
        print(f"  {ticker}: target={calib['target_breach_rate']:.2%}  "
              f"actual={calib['actual_breach_rate']:.2%}  n={calib['n_obs']}  "
              f"calibrated={calib['well_calibrated']}")

    print("\n=== Backtesting portfolios ===")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns[log_returns.index > split_date]

    results = {}
    for source, col in [("ML (LSTM)", "ml_pred"), ("GARCH", "garch_fwd_rvol"), ("EWMA", "ewma_fwd_rvol")]:
        fc = merged.reset_index()[["date", "ticker", col]].rename(columns={col: "ml_pred"}).dropna()
        fc = fc.set_index("date")
        if fc["ticker"].nunique() < 2:
            continue
        try:
            port_ret, weights = backtest_portfolio(log_returns, fc, forecast_col="ml_pred", scheme="risk_parity")
            results[source] = (port_ret, performance_summary(port_ret))
        except ValueError as e:
            print(f"  Skipping {source}: {e}")

    equal_weight = log_returns[tickers].mean(axis=1)
    results["Equal-weight"] = (equal_weight, performance_summary(equal_weight))

    print("\n--- Portfolio performance (risk-parity allocation) ---")
    for name, (_, stats) in results.items():
        print(f"  {name:15s}  ann_return={stats['annualized_return']:.2%}  "
              f"ann_vol={stats['annualized_vol']:.2%}  sharpe={stats['sharpe']:.2f}  "
              f"max_dd={stats['max_drawdown']:.2%}")

    print("\n=== Saving plots ===")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("LSTM training curve (point forecast)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()

    for name, (port_ret, _) in results.items():
        cum = (1 + port_ret).cumprod()
        axes[1].plot(cum.index, cum.values, label=name)
    axes[1].set_title("Cumulative portfolio returns (out-of-sample)")
    axes[1].legend()
    axes[1].set_ylabel("Growth of $1")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/results.png", dpi=120)
    print(f"Saved plot to {out_dir}/results.png")

    leaderboard.to_csv(f"{out_dir}/forecast_leaderboard.csv", index=False)
    print(f"Saved leaderboard to {out_dir}/forecast_leaderboard.csv")

    return leaderboard, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no internet needed)")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM", "XOM", "PG"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    run(args.tickers, args.demo, args.start, args.end, out_dir=args.out_dir)
