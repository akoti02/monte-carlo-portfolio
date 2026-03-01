"""Page 2: Monte Carlo Simulator — GBM and GARCH side by side."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from core.monte_carlo import (
    simulate_gbm,
    fit_garch_and_simulate,
    simulation_statistics,
)

st.set_page_config(page_title="Monte Carlo Simulator", page_icon="🎲", layout="wide")
st.title("Monte Carlo Simulator")

if "returns" not in st.session_state:
    st.warning("Please fetch data on the **Overview & Metrics** page first.")
    st.stop()

returns = st.session_state["returns"]
tickers = st.session_state["tickers"]
n = len(tickers)
weights = st.session_state.get("optimised_weights", np.ones(n) / n)
initial_value = st.session_state.get("initial_value", 100_000)

# --- Configuration ---
st.sidebar.header("Simulation Settings")
n_sims = st.sidebar.select_slider(
    "Number of Simulations", options=[1000, 5000, 10000, 25000, 50000],
    value=st.session_state.get("n_simulations", 10000),
)
n_days = st.sidebar.slider("Forecast Horizon (trading days)", 63, 504, 252)
st.session_state["n_simulations"] = n_sims

run_btn = st.sidebar.button("Run Simulations", type="primary", use_container_width=True)

if run_btn or "gbm_paths" in st.session_state:
    if run_btn:
        col_status = st.columns(2)
        with col_status[0]:
            with st.spinner("Running GBM simulation..."):
                gbm_paths = simulate_gbm(returns, weights, n_sims, n_days, initial_value)
                st.session_state["gbm_paths"] = gbm_paths
        with col_status[1]:
            with st.spinner("Running GARCH(1,1) simulation..."):
                try:
                    garch_paths = fit_garch_and_simulate(returns, weights, n_sims, n_days, initial_value)
                    st.session_state["garch_paths"] = garch_paths
                except Exception as e:
                    st.error(f"GARCH fitting failed: {e}. Using GBM results only.")
                    st.session_state["garch_paths"] = None

    gbm_paths = st.session_state["gbm_paths"]
    garch_paths = st.session_state.get("garch_paths")

    gbm_stats = simulation_statistics(gbm_paths)
    st.session_state["gbm_stats"] = gbm_stats

    # --- Side-by-side path plots ---
    st.subheader("Simulation Paths")

    def make_path_fig(paths, title):
        fig = go.Figure()
        n_show = min(300, paths.shape[0])
        days = np.arange(paths.shape[1])
        for i in range(n_show):
            fig.add_trace(go.Scatter(
                x=days, y=paths[i], mode="lines",
                line=dict(width=0.3, color="rgba(59,130,246,0.05)"),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=days, y=np.median(paths, axis=0), mode="lines",
            name="Median", line=dict(width=2.5, color="#ef4444"),
        ))
        fig.add_trace(go.Scatter(
            x=days, y=np.percentile(paths, 5, axis=0), mode="lines",
            name="5th Percentile", line=dict(width=1.5, color="#f59e0b", dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=days, y=np.percentile(paths, 95, axis=0), mode="lines",
            name="95th Percentile", line=dict(width=1.5, color="#10b981", dash="dash"),
        ))
        fig.update_layout(
            title=title, xaxis_title="Trading Days", yaxis_title="Portfolio Value ($)",
            height=450, template="plotly_white",
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_path_fig(gbm_paths, "GBM Simulation"), use_container_width=True)
    with col2:
        if garch_paths is not None:
            st.plotly_chart(make_path_fig(garch_paths, "GARCH(1,1) Simulation"), use_container_width=True)
        else:
            st.info("GARCH simulation not available.")

    # --- Distribution comparison ---
    st.subheader("Final Value Distributions")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=gbm_paths[:, -1], nbinsx=100, name="GBM",
        opacity=0.6, marker_color="#3b82f6",
    ))
    if garch_paths is not None:
        garch_stats = simulation_statistics(garch_paths)
        st.session_state["garch_stats"] = garch_stats
        fig_dist.add_trace(go.Histogram(
            x=garch_paths[:, -1], nbinsx=100, name="GARCH(1,1)",
            opacity=0.6, marker_color="#ef4444",
        ))
    fig_dist.update_layout(
        barmode="overlay", title="Distribution of Final Portfolio Values",
        xaxis_title="Final Value ($)", yaxis_title="Frequency",
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Statistics tables ---
    st.subheader("Simulation Statistics")
    stat_cols = st.columns(2)

    def stats_to_df(stats, label):
        rows = []
        for k, v in stats.items():
            if "Value" in k or "Percentile" in k or "VaR" in k or "CVaR" in k:
                rows.append({"Metric": k, label: f"${v:,.0f}"})
            elif "Prob" in k:
                rows.append({"Metric": k, label: f"{v:.2%}"})
            else:
                rows.append({"Metric": k, label: f"{v:.4f}"})
        return pd.DataFrame(rows).set_index("Metric")

    with stat_cols[0]:
        st.markdown("**GBM Statistics**")
        st.dataframe(stats_to_df(gbm_stats, "GBM"), use_container_width=True)
    with stat_cols[1]:
        if garch_paths is not None:
            st.markdown("**GARCH(1,1) Statistics**")
            st.dataframe(stats_to_df(garch_stats, "GARCH"), use_container_width=True)

else:
    st.info("Click **Run Simulations** in the sidebar to start.")
