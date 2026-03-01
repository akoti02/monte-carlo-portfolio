"""Page 5: Report Download — generate and download comprehensive PDF."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np

from core.report import generate_pdf_report
from core.backtest import backtest_portfolio, comparison_table

st.set_page_config(page_title="Report Download", page_icon="📄", layout="wide")
st.title("Report Download")

# --- Check what data is available ---
has_data = "returns" in st.session_state
has_mc = "gbm_paths" in st.session_state
has_opt = "frontier" in st.session_state
has_stress = "stress_results" in st.session_state
has_backtest = "backtest_result" in st.session_state

st.markdown("### Report Contents")
status_items = {
    "Portfolio Data & Metrics": has_data,
    "Monte Carlo Simulations (GBM)": has_mc,
    "Monte Carlo Simulations (GARCH)": st.session_state.get("garch_paths") is not None,
    "Portfolio Optimisation": has_opt,
    "Stress Test Results": has_stress,
    "Backtest Results": has_backtest,
}

for item, ready in status_items.items():
    icon = "✅" if ready else "⬜"
    st.markdown(f"{icon} {item}")

if not has_data:
    st.warning("Please run at least the **Overview & Metrics** page before generating a report.")
    st.stop()

st.markdown("---")

# --- Backtest comparison table ---
if has_backtest:
    st.subheader("Backtest Comparison")
    bt = st.session_state["backtest_result"]
    comp = comparison_table(bt)
    st.dataframe(comp, use_container_width=True)

# --- Generate Report ---
if st.button("Generate PDF Report", type="primary", use_container_width=True):
    with st.spinner("Generating PDF report..."):
        tickers = st.session_state["tickers"]
        n = len(tickers)
        weights = st.session_state.get("optimised_weights", np.ones(n) / n)
        metrics = st.session_state.get("metrics", {})

        pdf_bytes = generate_pdf_report(
            tickers=tickers,
            weights=weights,
            metrics=metrics,
            gbm_paths=st.session_state.get("gbm_paths"),
            garch_paths=st.session_state.get("garch_paths"),
            gbm_stats=st.session_state.get("gbm_stats"),
            garch_stats=st.session_state.get("garch_stats"),
            frontier=st.session_state.get("frontier"),
            stress_results=st.session_state.get("stress_results"),
            backtest_result=st.session_state.get("backtest_result"),
        )

    st.success("Report generated successfully!")
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="portfolio_risk_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
