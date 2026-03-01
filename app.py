"""Monte Carlo Portfolio Risk Analysis — Main Streamlit App."""

import streamlit as st

st.set_page_config(
    page_title="Monte Carlo Portfolio Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared session state defaults
if "tickers" not in st.session_state:
    st.session_state["tickers"] = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM"]
if "years" not in st.session_state:
    st.session_state["years"] = 5
if "n_simulations" not in st.session_state:
    st.session_state["n_simulations"] = 10_000
if "initial_value" not in st.session_state:
    st.session_state["initial_value"] = 100_000

st.title("Monte Carlo Portfolio Risk Analysis")
st.markdown(
    """
    Welcome to the **Monte Carlo Portfolio Risk Analysis Platform**. Use the sidebar
    to navigate between pages:

    - **Overview & Metrics** — Configure your portfolio, view historical returns, and key risk/return metrics
    - **Monte Carlo Simulator** — Run 10,000 simulations using GBM and GARCH(1,1) models side by side
    - **Portfolio Optimiser** — Find the efficient frontier and optimal weights via Markowitz optimisation
    - **Stress Test Results** — Replay your portfolio through historical crisis periods
    - **Report Download** — Generate and download a comprehensive PDF report

    ---
    **Getting Started:** Head to the **Overview & Metrics** page to configure your portfolio tickers
    and fetch data before running any analysis.
    """
)
