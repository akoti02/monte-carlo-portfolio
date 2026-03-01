"""Page 1: Overview & Metrics — configure portfolio and view key statistics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.data import fetch_stock_data, compute_returns, fetch_benchmark
from core.metrics import compute_all_metrics, portfolio_returns

st.set_page_config(page_title="Overview & Metrics", page_icon="📈", layout="wide")
st.title("Overview & Metrics")

# --- Sidebar: Portfolio Configuration ---
st.sidebar.header("Portfolio Configuration")

ticker_input = st.sidebar.text_input(
    "Tickers (comma-separated, 5–10)",
    value=", ".join(st.session_state.get("tickers", ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM"])),
)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if len(tickers) < 2 or len(tickers) > 15:
    st.sidebar.error("Enter between 2 and 15 tickers.")
    st.stop()

years = st.sidebar.slider("Years of history", 1, 10, st.session_state.get("years", 5))
initial_value = st.sidebar.number_input(
    "Initial Portfolio Value ($)", min_value=1000, value=st.session_state.get("initial_value", 100_000), step=1000,
)

st.session_state["tickers"] = tickers
st.session_state["years"] = years
st.session_state["initial_value"] = initial_value

# --- Fetch Data ---
fetch_btn = st.sidebar.button("Fetch Data", type="primary", use_container_width=True)

if fetch_btn or "prices" in st.session_state:
    if fetch_btn:
        with st.spinner("Fetching stock data..."):
            try:
                prices = fetch_stock_data(tickers, years)
                benchmark = fetch_benchmark(years)
                returns = compute_returns(prices)
                bench_returns = compute_returns(pd.DataFrame(benchmark)).iloc[:, 0]

                st.session_state["prices"] = prices
                st.session_state["returns"] = returns
                st.session_state["benchmark"] = benchmark
                st.session_state["bench_returns"] = bench_returns
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

    prices = st.session_state["prices"]
    returns = st.session_state["returns"]
    bench_returns = st.session_state["bench_returns"]

    # --- Portfolio weights (equal weight default until optimised) ---
    n = len(tickers)
    if "optimised_weights" in st.session_state:
        weights = st.session_state["optimised_weights"]
    else:
        weights = np.ones(n) / n
        st.session_state["weights"] = weights

    port_rets = portfolio_returns(returns, weights)
    metrics = compute_all_metrics(port_rets, bench_returns)
    st.session_state["metrics"] = metrics

    # --- Display ---
    st.subheader("Price History")
    normalised = prices / prices.iloc[0] * 100
    fig = px.line(normalised, title="Normalised Prices (base = 100)")
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics cards
    st.subheader("Key Metrics")
    cols = st.columns(5)
    metric_items = list(metrics.items())
    for i, (k, v) in enumerate(metric_items):
        col = cols[i % 5]
        if isinstance(v, float):
            if abs(v) < 2 and k not in ("Sharpe Ratio", "Sortino Ratio", "Beta vs S&P500"):
                col.metric(k, f"{v:.2%}")
            else:
                col.metric(k, f"{v:.4f}")

    # Correlation heatmap
    st.subheader("Return Correlation Matrix")
    corr = returns.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Pairwise Correlations",
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Return distribution
    st.subheader("Portfolio Return Distribution")
    fig_hist = px.histogram(
        port_rets, nbins=100, title="Daily Return Distribution",
        labels={"value": "Daily Return", "count": "Frequency"},
    )
    fig_hist.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Current weights
    st.subheader("Current Portfolio Weights")
    weight_type = "Optimised" if "optimised_weights" in st.session_state else "Equal"
    st.caption(f"Allocation type: **{weight_type}**")
    w_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    fig_pie = px.pie(w_df, names="Ticker", values="Weight", title="Allocation")
    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Configure your portfolio in the sidebar and click **Fetch Data** to begin.")
