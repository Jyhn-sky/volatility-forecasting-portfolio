# Live Portfolio Tracker & Dashboard

A Python tracker for a mock portfolio of 5–10 stocks. Every buy and sell is
logged with the reasoning behind it, and a live dashboard measures how
those decisions perform against a benchmark (S&P 500 by default, or the
S&P/TSX).

The point is to show ongoing discipline: every position has a written
thesis, a catalyst, a key risk and an exit trigger, and the numbers show
honestly whether those decisions beat the market.

## What it shows

- **Performance:** portfolio value, total return, and return vs the benchmark
- **Risk-adjusted:** Jensen's **alpha**, **beta**, correlation, volatility,
  Sharpe ratio
- **Drawdown:** max and current drop from peak, for both portfolio and benchmark
- **Sector allocation** (cash included)
- **Charts:** growth of 100 vs the benchmark, drawdown over time, sector pie
- **Holdings:** live price, average cost, market value, weight, unrealized P/L
- **Trade journal:** every decision with its thesis, catalyst, risk and exit trigger

## Project structure

```
app.py               # Streamlit dashboard
report.py            # same numbers printed in the terminal (--save writes CSVs)
config.toml          # start date, starting cash, benchmark, risk-free rate, sectors
data/trades.csv      # the trade journal: what you bought/sold and why
tracker/
    data.py          # loads config + trades, pulls prices from Yahoo Finance
    portfolio.py     # trades -> daily positions, cash and value; holdings table
    metrics.py       # returns, alpha, beta, volatility, Sharpe, drawdown, sectors
    analysis.py      # ties it together for the dashboard and report
tests/               # offline tests using made-up prices
.vscode/             # run/debug configs, test settings, recommended extensions
```

## Setup (VS Code)

1. Open the `portfolio-tracker` folder in VS Code (**File → Open Folder**),
   so the `.vscode` settings load. Install the recommended Python extension
   when VS Code prompts you.
2. Create a virtual environment and install the requirements:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Then pick `.venv` as the interpreter (**Ctrl/Cmd+Shift+P → Python: Select Interpreter**).
3. Run it from the **Run and Debug** panel (Ctrl/Cmd+Shift+D):
   - **Dashboard (Streamlit)** opens the dashboard in your browser
   - **Terminal report** prints the summary in the terminal

Or from the terminal:

```bash
streamlit run app.py        # dashboard at http://localhost:8501
python report.py            # terminal summary
python report.py --save     # also writes output/history.csv, holdings.csv, summary.csv
pytest                      # tests (no internet needed)
```

Needs Python 3.11+ and an internet connection, since prices come from
Yahoo Finance via yfinance.

## Using it

1. **`config.toml`:** set your start date, starting cash and benchmark
   (`SPY` for the S&P 500, `XIC.TO` for the S&P/TSX), and a sector for
   each ticker.
2. **`data/trades.csv`:** add a row for every buy and sell. Leave `price`
   blank to use that day's closing price, or fill in your real fill price.
   Write the thesis *before* you know how the trade turns out.
3. Open the dashboard weekly. When an exit trigger is hit, act on it and
   log the trade with the reason.
4. Commit `trades.csv` each time you trade. The git history becomes a
   timestamped record that you made each call when you said you did.

The example portfolio (8 stocks across 8 sectors, opened 2026-01-02) and
its theses are placeholders. Replace them with your own picks and reasoning.

## How the numbers are calculated

- **Daily portfolio value** = cash + Σ (shares held that day × closing
  price). Trades fill at the close, so buying or selling moves money
  between cash and stock without creating a fake jump in returns.
- **Beta** = covariance of daily portfolio and benchmark returns ÷
  benchmark variance.
- **Jensen's alpha** = (avg daily return − rf) − β × (avg benchmark daily
  return − rf), × 252 to annualize.
- **Volatility** = stdev of daily returns × √252. **Sharpe** = (annualized
  return − rf) ÷ volatility.
- **Drawdown** = value ÷ running peak − 1.
- **Average cost** = weighted average of all buys. Selling doesn't change
  it, and realized gains from sells sit in cash.

## Limitations

- Price returns only; dividends aren't counted, so income-heavy stocks look
  slightly worse than they really did.
- Yahoo Finance quotes can be ~15 minutes delayed and occasionally fail to
  load; hit **Refresh prices** in the sidebar.
- Keep all holdings and the benchmark in one currency (all USD or all CAD),
  or the comparison is meaningless.
- Alpha and beta are noisy with only a few months of data. Treat them as
  rough until you have at least 6–12 months of history.
