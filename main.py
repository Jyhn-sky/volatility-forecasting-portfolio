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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline import fetch_real_data, generate_synthetic_data, compute_features
from baseline_models import rolling_baseline_forecasts
from ml_model import train_model, predict, FEATURE_COLS
from backtest import backtest_portfolio, performance_summary, regime_performance
from evaluate import compare_forecasts, var_calibration

warnings.filterwarnings("ignore")

# Default regime windows for breaking down backtest performance. These are
# tuned to 2022-2023 US market history (rate-hike selloff, then the March
# 2023 regional banking crisis, then recovery) since that's the period this
# project has been validated against with bank-stock tickers. If you're
# backtesting a different date range or asset class, edit these to match
# whatever regimes are actually relevant to your own test window --
# sub-periods that don't overlap your data are skipped automatically.
DEFAULT_REGIMES = [
    ("Rate-hike selloff", "2022-01-01", "2022-10-15"),
    ("2023 banking crisis", "2023-03-01", "2023-05-15"),
    ("Recovery", "2023-05-16", "2023-12-31"),
]


def plot_per_ticker(merged, prices, tickers, split_date, out_dir, save=True):
    """
    For each ticker, build a two-panel chart:
      - top: the stock's own price performance during the test window
      - bottom: each model's forward-volatility forecast vs. what actually
        happened (the realized volatility), over time

    This is the "look at one company specifically" view -- useful when you
    care about a stock's own risk profile rather than a multi-asset
    portfolio decision.

    If save=True, each chart is written to disk as a PNG. If save=False,
    the figure is left open (not closed, not written to disk) so it can be
    displayed all at once later via plt.show() -- nothing touches the
    filesystem in that mode.
    """
    saved = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        g = merged[merged["ticker"] == ticker].sort_index()
        if g.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(f"{ticker} volatility")

        test_prices = prices.loc[prices.index > split_date, ticker].dropna()
        if len(test_prices) > 0:
            norm = test_prices / test_prices.iloc[0]
            axes[0].plot(norm.index, norm.values, color="black")
        axes[0].set_title(f"{ticker}: price performance (out-of-sample)")
        axes[0].set_ylabel("Growth of $1")
        axes[0].axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

        axes[1].plot(g.index, g["target_fwd_rvol"], label="Realized (actual)",
                     color="black", linewidth=2)
        for col, label in [("ewma_fwd_rvol", "EWMA"), ("garch_fwd_rvol", "GARCH"),
                           ("ml_pred", "LSTM")]:
            if col in g.columns and g[col].notna().any():
                axes[1].plot(g.index, g[col], label=label, alpha=0.85)
        axes[1].set_title(f"{ticker}: forward volatility forecasts vs. realized")
        axes[1].set_ylabel("Annualized volatility")
        axes[1].legend()

        fig.tight_layout()

        if save:
            fname = f"{out_dir}/{ticker}_volatility.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            saved.append(fname)
            print(f"Saved per-ticker plot to {fname}")
        # else: leave the figure open, don't touch disk -- shown later by plt.show()

    return saved


def run(tickers, demo, start, end, train_frac=0.7, out_dir=".", save=True):
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

    print("\n=== Forecast accuracy: ML vs GARCH vs EWMA (all tickers combined) ===")
    leaderboard = compare_forecasts(merged, forecast_cols=("ewma_fwd_rvol", "garch_fwd_rvol", "ml_pred"))
    print(leaderboard.to_string(index=False))

    print("\n=== Forecast accuracy per ticker ===")
    for ticker, g in merged.groupby("ticker"):
        per_ticker_board = compare_forecasts(g, forecast_cols=("ewma_fwd_rvol", "garch_fwd_rvol", "ml_pred"))
        print(f"\n  -- {ticker} --")
        print("  " + per_ticker_board.to_string(index=False).replace("\n", "\n  "))

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

    print("\n--- Regime breakdown (does performance hold up across different market conditions?) ---")
    regime_tables = {}
    for name, (port_ret, _) in results.items():
        rt = regime_performance(port_ret, DEFAULT_REGIMES)
        regime_tables[name] = rt
        if len(rt) <= 1:  # only the "Full period" row matched -- none of the named regimes overlapped
            continue
        print(f"\n  {name}:")
        for _, row in rt.iterrows():
            print(f"    {row['regime']:22s} n={row['n_days']:4d}  "
                  f"sharpe={row['sharpe']:6.2f}  max_dd={row['max_drawdown']:7.2%}  "
                  f"ann_return={row['annualized_return']:7.2%}")

    if any(len(rt) > 1 for rt in regime_tables.values()):
        combined = pd.concat(
            [rt.assign(strategy=name) for name, rt in regime_tables.items()], ignore_index=True
        )
        if save:
            combined.to_csv(f"{out_dir}/regime_breakdown.csv", index=False)
            print(f"\nSaved regime breakdown to {out_dir}/regime_breakdown.csv")
    else:
        print("  (None of the default regime windows overlapped this backtest's date range --"
              " edit DEFAULT_REGIMES in main.py to match your own test period.)")

    print("\n=== Building plots ===")
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
    if save:
        fig.savefig(f"{out_dir}/results.png", dpi=120)
        print(f"Saved plot to {out_dir}/results.png")

    if save:
        leaderboard.to_csv(f"{out_dir}/forecast_leaderboard.csv", index=False)
        print(f"Saved leaderboard to {out_dir}/forecast_leaderboard.csv")

    print("\n=== Building per-ticker volatility plots ===")
    per_ticker_plots = plot_per_ticker(merged, prices, tickers, split_date, out_dir, save=save)

    if not save:
        print("\n--save was not passed, so nothing was written to disk.")
        print("Close the chart windows to end the program.")
        plt.show()  # blocks until all open figure windows are closed; nothing is saved

    return leaderboard, results, per_ticker_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no internet needed)")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM", "XOM", "PG"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--save", action="store_true",
                         help="Write results.png, per-ticker PNGs, and CSVs to disk. "
                              "Without this flag, charts open in windows and nothing is "
                              "saved -- just close the windows when you're done looking.")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")  # no display needed, just write files
    else:
        try:
            matplotlib.use("TkAgg")  # interactive window backend, ships with most Python installs
        except Exception:
            print("Could not open an interactive plot window on this system "
                  "(no display / Tk not available). Falling back to --save mode instead.")
            matplotlib.use("Agg")
            args.save = True

    run(args.tickers, args.demo, args.start, args.end, out_dir=args.out_dir, save=args.save)
