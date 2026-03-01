"""Page 4: Stress Test Results — replay through historical crises."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.stress_test import run_all_stress_tests, stress_test_summary_df, CRISIS_PERIODS
from core.data import fetch_stock_data, compute_returns

st.set_page_config(page_title="Stress Test Results", page_icon="🔥", layout="wide")
st.title("Stress Test Results")

if "returns" not in st.session_state:
    st.warning("Please fetch data on the **Overview & Metrics** page first.")
    st.stop()

tickers = st.session_state["tickers"]
n = len(tickers)
weights = st.session_state.get("optimised_weights", np.ones(n) / n)

st.sidebar.header("Stress Test Configuration")
st.sidebar.markdown("Crisis periods are replayed using actual historical returns.")
for name, (start, end) in CRISIS_PERIODS.items():
    st.sidebar.markdown(f"- **{name}**: {start} to {end}")

run_btn = st.sidebar.button("Run Stress Tests", type="primary", use_container_width=True)

if run_btn or "stress_results" in st.session_state:
    if run_btn:
        with st.spinner("Fetching extended history and running stress tests..."):
            try:
                # Fetch longer history to cover all crises
                prices_long = fetch_stock_data(tickers, years=20)
                returns_long = compute_returns(prices_long)
                results = run_all_stress_tests(returns_long, weights)
                st.session_state["stress_results"] = results
                st.session_state["returns_long"] = returns_long
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Some tickers may not have data going back to 2007. Try using tickers with longer histories.")
                st.stop()

    results = st.session_state["stress_results"]

    if not results:
        st.warning("No crisis periods are covered by the available data.")
        st.stop()

    # --- Summary Table ---
    st.subheader("Stress Test Summary")
    summary = stress_test_summary_df(results)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # --- Individual Crisis Charts ---
    st.subheader("Crisis Period Performance")

    for r in results:
        st.markdown(f"### {r['name']}")
        col1, col2 = st.columns([3, 1])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=r["cumulative"].index,
                y=r["cumulative"].values,
                mode="lines",
                name="Portfolio",
                line=dict(color="#ef4444", width=2),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.1)",
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="grey", annotation_text="Break-even")
            fig.update_layout(
                title=f"{r['name']} — Cumulative Return",
                xaxis_title="Date", yaxis_title="Cumulative Return (1 = initial)",
                height=350, template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Total Return", f"{r['cumulative_return']:.2%}")
            st.metric("Max Drawdown", f"{r['max_drawdown']:.2%}")
            st.metric("Annualised Vol", f"{r['annualised_vol']:.2%}")
            st.metric("Worst Day", f"{r['worst_day']:.2%}")
            st.metric("Best Day", f"{r['best_day']:.2%}")

    # --- Comparison bar chart ---
    st.subheader("Crisis Impact Comparison")
    crisis_names = [r["name"] for r in results]
    cum_rets = [r["cumulative_return"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]

    fig_comp = make_subplots(rows=1, cols=2, subplot_titles=["Cumulative Return", "Max Drawdown"])
    fig_comp.add_trace(
        go.Bar(x=crisis_names, y=cum_rets, marker_color="#3b82f6", name="Return"),
        row=1, col=1,
    )
    fig_comp.add_trace(
        go.Bar(x=crisis_names, y=drawdowns, marker_color="#ef4444", name="Drawdown"),
        row=1, col=2,
    )
    fig_comp.update_layout(
        height=350, template="plotly_white", showlegend=False,
    )
    fig_comp.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("Click **Run Stress Tests** in the sidebar to begin.")
