"""
ml_model.py

LSTM-based forward volatility forecaster. Two training modes:
  - "point": MSE loss on log-realized-vol (comparable to GARCH/EWMA)
  - "quantile": pinball loss at a chosen quantile (e.g. 0.05), which gives
    you a direct VaR-style estimate with a proper calibration check
    (see evaluate.py) instead of just a point forecast.

Feature scaling and sequence construction are handled here so training
code stays clean.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

FEATURE_COLS = ["rvol_5d", "rvol_10d", "rvol_21d", "rvol_63d", "mom_21d", "mom_63d", "vix", "vix_chg_5d"]
SEQ_LEN = 21


class VolSequenceDataset(Dataset):
    """Builds sliding-window sequences per ticker (no cross-ticker leakage)."""

    def __init__(self, panel: pd.DataFrame, feature_cols=FEATURE_COLS, seq_len=SEQ_LEN,
                 target_col="target_fwd_rvol", scaler=None, fit_scaler=False):
        self.seq_len = seq_len
        self.feature_cols = feature_cols

        if fit_scaler:
            self.mean = panel[feature_cols].mean()
            self.std = panel[feature_cols].std().replace(0, 1.0)
        else:
            assert scaler is not None, "Must supply a fitted scaler when fit_scaler=False"
            self.mean, self.std = scaler

        X, y = [], []
        for _, g in panel.groupby("ticker"):
            g = g.sort_index()
            feats = ((g[feature_cols] - self.mean) / self.std).values
            targets = g[target_col].values
            for i in range(seq_len, len(g)):
                X.append(feats[i - seq_len:i])
                y.append(targets[i])

        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    def scaler(self):
        return (self.mean, self.std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class VolLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1, dropout=0.1, nonneg_output=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        layers = [nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1)]
        if nonneg_output:
            layers.append(nn.Softplus())  # keeps volatility forecasts non-negative
        # nonneg_output=False for return-quantile models, since a 5% quantile
        # of daily returns is a negative number by construction
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


def pinball_loss(y_pred, y_true, quantile):
    diff = y_true - y_pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1) * diff))


def train_model(train_panel, val_panel, mode="point", quantile=0.05,
                 target_col=None, epochs=30, batch_size=64, lr=1e-3,
                 hidden_size=16, weight_decay=1e-4, patience=5, verbose=True):
    """
    Train a VolLSTM.
      mode="point"    -> MSE against target_fwd_rvol (forward realized vol;
                          comparable to GARCH/EWMA).
      mode="quantile" -> pinball loss against target_next_return (the
                          model directly predicts a lower quantile of the
                          *return* distribution -- this is what makes it
                          an actual VaR forecast, not a vol forecast in
                          disguise). Softplus output is disabled for this
                          mode since return quantiles are negative.

    Regularization / early stopping: validation loss on this kind of panel
    tends to bottom out within the first few epochs and then climb back up
    as the model starts memorizing the training window (classic overfitting
    -- we saw this directly in early runs of this project). To address it:
      - hidden_size defaults to 16 instead of 32 (less capacity to overfit)
      - weight_decay adds L2 regularization via the optimizer
      - training stops early once val_loss hasn't improved for `patience`
        epochs, and the model's weights are rolled back to whichever epoch
        had the best validation loss (not just whatever epoch training
        happened to stop at)
    Returns (model, scaler, history).
    """
    if target_col is None:
        target_col = "target_next_return" if mode == "quantile" else "target_fwd_rvol"

    train_ds = VolSequenceDataset(train_panel, target_col=target_col, fit_scaler=True)
    val_ds = VolSequenceDataset(val_panel, target_col=target_col, scaler=train_ds.scaler(), fit_scaler=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VolLSTM(n_features=len(FEATURE_COLS), hidden_size=hidden_size, nonneg_output=(mode == "point"))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            if mode == "point":
                loss = nn.functional.mse_loss(pred, yb)
            else:
                loss = pinball_loss(pred, yb, quantile)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = (nn.functional.mse_loss(pred, yb) if mode == "point"
                        else pinball_loss(pred, yb, quantile))
                val_losses.append(loss.item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        current_val_loss = history["val_loss"][-1]

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train_loss={history['train_loss'][-1]:.5f}  "
                  f"val_loss={history['val_loss'][-1]:.5f}")

        if epochs_since_improvement >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs; "
                      f"best val_loss={best_val_loss:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)  # roll back to the best-val-loss epoch,
        # not whatever epoch training happened to stop at

    return model, train_ds.scaler(), history


def predict(model, panel, scaler, feature_cols=FEATURE_COLS, seq_len=SEQ_LEN, target_col="target_fwd_rvol"):
    """Generate predictions for every valid sequence in `panel`, returned
    aligned to the panel's (ticker, date) so they can be merged with
    the baseline forecasts for comparison. `target_col` only needs to be
    a real column so the dataset can build (it's not used for the output)."""
    ds = VolSequenceDataset(panel, feature_cols, seq_len, target_col=target_col, scaler=scaler, fit_scaler=False)
    model.eval()
    with torch.no_grad():
        preds = model(ds.X).squeeze(-1).numpy()

    # rebuild the (ticker, date) index in the same order the dataset built sequences
    idx_records = []
    for ticker, g in panel.groupby("ticker"):
        g = g.sort_index()
        for i in range(seq_len, len(g)):
            idx_records.append((ticker, g.index[i]))

    out = pd.DataFrame(idx_records, columns=["ticker", "date"]).set_index("date")
    out["ml_pred"] = preds
    return out


if __name__ == "__main__":
    from data_pipeline import generate_synthetic_data, compute_features

    tickers = ["AAPL", "MSFT", "JPM"]
    prices, vix = generate_synthetic_data(tickers, n_days=1200)
    panel = compute_features(prices, vix)

    dates = np.sort(panel.index.unique())
    split = dates[int(len(dates) * 0.8)]
    train_panel = panel[panel.index <= split]
    val_panel = panel[panel.index > split]

    model, scaler, history = train_model(train_panel, val_panel, mode="point", epochs=15)
    preds = predict(model, val_panel, scaler)
    print(preds.head())
