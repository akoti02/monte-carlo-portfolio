"""Page 3: Portfolio Optimiser — Markowitz efficient frontier."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.optimizer import (
    optimise_max_sharpe,
    optimise_min_variance,
    compute_efficient_frontier,
    generate_random_portfolios,
)
from core.backtest import backtest_portfolio
from core.metrics import portfolio_returns, compute_all_metrics

st.set_page_config(page_title="Portfolio Optimiser", page_icon="⚡", layout="wide")
st.title("Portfolio Optimiser")

if "returns" not in st.session_state:
    st.warning("Please fetch data on the **Overview & Metrics** page first.")
    st.stop()

returns = st.session_state["returns"]
tickers = st.session_state["tickers"]
bench_returns = st.session_state.get("bench_returns")

run_btn = st.sidebar.button("Run Optimisation", type="primary", use_container_width=True)

if run_btn or "frontier" in st.session_state:
    if run_btn:
        with st.spinner("Computing efficient frontier..."):
            frontier = compute_efficient_frontier(returns, n_points=80)
            rand_ports = generate_random_portfolios(returns, 5000)
            st.session_state["frontier"] = frontier
            st.session_state["rand_ports"] = rand_ports

            # Store optimised weights for use across pages
            opt_w = frontier["max_sharpe"]["weights"]
            st.session_state["optimised_weights"] = opt_w
            st.session_state["weights"] = opt_w

            # Run backtest
            bt = backtest_portfolio(returns, opt_w, bench_returns)
            st.session_state["backtest_result"] = bt

    frontier = st.session_state["frontier"]
    rand_ports = st.session_state["rand_ports"]
    max_sharpe = frontier["max_sharpe"]
    min_var = frontier["min_variance"]

    # --- Efficient Frontier Plot ---
    st.subheader("Efficient Frontier")
    fig = go.Figure()

    # Random portfolios
    fig.add_trace(go.Scatter(
        x=rand_ports["volatilities"], y=rand_ports["returns"],
        mode="markers", name="Random Portfolios",
        marker=dict(
            size=3, color=rand_ports["sharpe_ratios"],
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="Sharpe"),
        ),
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    ))

    # Frontier line
    fig.add_trace(go.Scatter(
        x=frontier["volatilities"], y=frontier["returns"],
        mode="lines", name="Efficient Frontier",
        line=dict(color="#ef4444", width=3),
    ))

    # Max Sharpe point
    fig.add_trace(go.Scatter(
        x=[max_sharpe["volatility"]], y=[max_sharpe["return"]],
        mode="markers", name=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
        marker=dict(size=15, color="#f59e0b", symbol="star"),
    ))

    # Min Variance point
    fig.add_trace(go.Scatter(
        x=[min_var["volatility"]], y=[min_var["return"]],
        mode="markers", name="Min Variance",
        marker=dict(size=12, color="#10b981", symbol="diamond"),
    ))

    fig.update_layout(
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"),
        height=550, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Optimal Weights ---
    st.subheader("Optimal Portfolio Weights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Max Sharpe Ratio Portfolio**")
        w_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": max_sharpe["weights"],
        })
        w_df["Weight %"] = w_df["Weight"].apply(lambda x: f"{x:.2%}")
        st.dataframe(w_df[["Ticker", "Weight %"]], use_container_width=True, hide_index=True)
        st.metric("Expected Return", f"{max_sharpe['return']:.2%}")
        st.metric("Expected Volatility", f"{max_sharpe['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.4f}")

    with col2:
        st.markdown("**Min Variance Portfolio**")
        w_df2 = pd.DataFrame({
            "Ticker": tickers,
            "Weight": min_var["weights"],
        })
        w_df2["Weight %"] = w_df2["Weight"].apply(lambda x: f"{x:.2%}")
        st.dataframe(w_df2[["Ticker", "Weight %"]], use_container_width=True, hide_index=True)
        st.metric("Expected Return", f"{min_var['return']:.2%}")
        st.metric("Expected Volatility", f"{min_var['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{min_var['sharpe']:.4f}")

    # --- Weight comparison bar chart ---
    st.subheader("Weight Comparison")
    w_compare = pd.DataFrame({
        "Ticker": tickers * 3,
        "Weight": np.concatenate([
            max_sharpe["weights"],
            min_var["weights"],
            np.ones(len(tickers)) / len(tickers),
        ]),
        "Strategy": (
            ["Max Sharpe"] * len(tickers)
            + ["Min Variance"] * len(tickers)
            + ["Equal Weight"] * len(tickers)
        ),
    })
    import plotly.express as px
    fig_bar = px.bar(
        w_compare, x="Ticker", y="Weight", color="Strategy",
        barmode="group", title="Portfolio Weight Comparison",
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Backtest Preview ---
    if "backtest_result" in st.session_state:
        st.subheader("Backtest Preview")
        bt = st.session_state["backtest_result"]
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt["optimised"]["cumulative"].index,
            y=bt["optimised"]["cumulative"].values,
            name="Optimised", line=dict(color="#3b82f6", width=2),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt["equal_weight"]["cumulative"].index,
            y=bt["equal_weight"]["cumulative"].values,
            name="Equal Weight", line=dict(color="#ef4444", width=2),
        ))
        fig_bt.update_layout(
            title="Cumulative Performance: Optimised vs Equal Weight",
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            height=400, template="plotly_white",
        )
        st.plotly_chart(fig_bt, use_container_width=True)

else:
    st.info("Click **Run Optimisation** in the sidebar to compute the efficient frontier.")
