"""
Streamlit dashboard.

Run:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tracker import metrics
from tracker.analysis import load_and_analyze

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.toml"
TRADES = ROOT / "data" / "trades.csv"

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")


@st.cache_data(ttl=300, show_spinner="Pulling prices from Yahoo Finance...")
def load(config_mtime, trades_mtime):
    # mtimes are only cache keys, so editing either file forces a reload
    return load_and_analyze(CONFIG, TRADES)


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:+.2%}"


def vol(x):
    return "n/a" if pd.isna(x) else f"{x:.2%}"


def num(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}"


with st.sidebar:
    st.header("Controls")
    if st.button("Refresh prices"):
        st.cache_data.clear()
    st.caption("Prices refresh automatically every 5 minutes. "
               "Edit config.toml and data/trades.csv to change the portfolio.")

try:
    r = load(CONFIG.stat().st_mtime, TRADES.stat().st_mtime)
except Exception as e:  # show the problem instead of a stack trace
    st.error(f"Couldn't build the portfolio: {e}")
    st.stop()

s, c = r.summary, r.config
st.title(c.name)
st.caption(f"Benchmark: {c.benchmark}  ·  Since {c.start_date.date()}  ·  "
           f"{s['Trading days']} trading days")

# ---- headline numbers
k = st.columns(4)
k[0].metric("Portfolio value", f"${s['Portfolio value']:,.0f}", pct(s["Total return"]))
k[1].metric("Excess return vs benchmark", pct(s["Excess return"]),
            f"Benchmark {pct(s['Benchmark return'])}", delta_color="off")
k[2].metric("Alpha (annual)", pct(s["Alpha (annual)"]))
k[3].metric("Beta", num(s["Beta"]))
k = st.columns(4)
k[0].metric("Sharpe ratio", num(s["Sharpe ratio"]))
k[1].metric("Volatility", vol(s["Volatility"]),
            f"Benchmark {vol(s['Benchmark volatility'])}", delta_color="off")
k[2].metric("Max drawdown", pct(s["Max drawdown"]),
            f"Benchmark {pct(s['Benchmark max drawdown'])}", delta_color="off")
k[3].metric("Current drawdown", pct(s["Current drawdown"]))

# ---- charts
left, right = st.columns([2, 1])
with left:
    growth = pd.DataFrame({
        "Portfolio": metrics.growth_of_100(r.history["value"]),
        c.benchmark: metrics.growth_of_100(r.benchmark),
    })
    fig = px.line(growth, labels={"value": "Growth of 100", "index": "", "variable": ""},
                  title="Portfolio vs benchmark (start = 100)")
    fig.update_layout(legend=dict(orientation="h", y=1.1), margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    dd = pd.DataFrame({
        "Portfolio": metrics.drawdown(r.history["value"]),
        c.benchmark: metrics.drawdown(r.benchmark),
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd["Portfolio"], fill="tozeroy", name="Portfolio"))
    fig.add_trace(go.Scatter(x=dd.index, y=dd[c.benchmark], name=c.benchmark,
                             line=dict(dash="dot")))
    fig.update_layout(title="Drawdown from peak", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=1.15), margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")

with right:
    fig = px.pie(values=r.sectors.values, names=r.sectors.index, hole=0.45,
                 title="Sector allocation")
    fig.update_traces(textinfo="percent", sort=False)
    fig.update_layout(margin=dict(t=60, b=10))
    st.plotly_chart(fig, width="stretch")

# ---- holdings
st.subheader("Holdings")
st.dataframe(
    r.holdings, hide_index=True, width="stretch",
    column_config={
        "ticker": "Ticker", "sector": "Sector",
        "shares": st.column_config.NumberColumn("Shares", format="%.0f"),
        "avg_cost": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "market_value": st.column_config.NumberColumn("Market value", format="$%,.0f"),
        "cost_basis": st.column_config.NumberColumn("Cost basis", format="$%,.0f"),
        "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L", format="$%+,.0f"),
        "return": st.column_config.NumberColumn("Return", format="percent"),
        "weight": st.column_config.NumberColumn("Weight", format="percent"),
    })
st.caption(f"Cash: ${r.cash:,.2f}")

# ---- decision journal
st.subheader("Trade journal")
journal = r.trades[["date", "ticker", "action", "shares", "price", "thesis",
                    "catalyst", "key_risk", "exit_trigger"]].sort_values("date", ascending=False)
st.dataframe(
    journal, hide_index=True, width="stretch",
    column_config={
        "date": st.column_config.DateColumn("Date"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "thesis": st.column_config.TextColumn("Thesis", width="large"),
        "key_risk": "Key risk", "exit_trigger": "Exit / review trigger",
    })
