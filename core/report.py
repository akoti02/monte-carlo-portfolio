"""PDF report generation using ReportLab."""

import io
import tempfile
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


def _fig_to_image(fig, width=6 * inch, height=3.5 * inch):
    """Convert matplotlib figure to ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def _make_table(data: list[list], col_widths=None) -> Table:
    """Create a styled ReportLab table."""
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
            ]
        )
    )
    return t


def _plot_simulation_paths(paths: np.ndarray, title: str):
    """Plot a sample of simulation paths."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_show = min(200, paths.shape[0])
    for i in range(n_show):
        ax.plot(paths[i], alpha=0.03, color="#3b82f6", linewidth=0.5)

    ax.plot(np.median(paths, axis=0), color="#ef4444", linewidth=2, label="Median")
    ax.plot(np.percentile(paths, 5, axis=0), "--", color="#f59e0b", linewidth=1.5, label="5th pctl")
    ax.plot(np.percentile(paths, 95, axis=0), "--", color="#10b981", linewidth=1.5, label="95th pctl")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_distribution(paths: np.ndarray, title: str):
    """Plot histogram of final portfolio values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    final = paths[:, -1]
    ax.hist(final, bins=80, alpha=0.7, color="#3b82f6", edgecolor="#1e40af")
    ax.axvline(np.median(final), color="#ef4444", linewidth=2, label=f"Median: ${np.median(final):,.0f}")
    ax.axvline(np.percentile(final, 5), color="#f59e0b", linewidth=1.5, linestyle="--", label=f"5th pctl: ${np.percentile(final, 5):,.0f}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Final Portfolio Value ($)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_backtest(backtest_result: dict):
    """Plot backtest cumulative performance."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    opt = backtest_result["optimised"]["cumulative"]
    ew = backtest_result["equal_weight"]["cumulative"]
    ax.plot(opt.index, opt.values, label="Optimised", color="#3b82f6", linewidth=1.5)
    ax.plot(ew.index, ew.values, label="Equal Weight", color="#ef4444", linewidth=1.5)
    ax.set_title("Backtest: Optimised vs Equal Weight", fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_pdf_report(
    tickers: list[str],
    weights: np.ndarray,
    metrics: dict,
    gbm_paths: np.ndarray | None = None,
    garch_paths: np.ndarray | None = None,
    gbm_stats: dict | None = None,
    garch_stats: dict | None = None,
    frontier: dict | None = None,
    stress_results: list[dict] | None = None,
    backtest_result: dict | None = None,
) -> bytes:
    """Generate a comprehensive PDF report and return as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=15 * mm, rightMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=22,
        textColor=colors.HexColor("#1f2937"), spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#1e40af"), spaceBefore=16, spaceAfter=8,
    )
    body_style = styles["BodyText"]

    elements = []

    # --- Title ---
    elements.append(Paragraph("Monte Carlo Portfolio Risk Analysis Report", title_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Portfolio: {', '.join(tickers)}", body_style
    ))
    elements.append(Spacer(1, 12))

    # --- Methodology ---
    elements.append(Paragraph("Methodology", h2_style))
    elements.append(Paragraph(
        "This report analyses portfolio risk using two Monte Carlo simulation approaches: "
        "<b>Geometric Brownian Motion (GBM)</b>, which assumes constant drift and volatility, "
        "and <b>GARCH(1,1)</b>, which captures time-varying volatility clustering observed in "
        "financial markets. 10,000 simulation paths are generated for each model over a 1-year "
        "forward horizon. Portfolio weights are optimised using <b>Markowitz mean-variance "
        "optimisation</b> to maximise the Sharpe ratio. Historical stress tests replay portfolio "
        "performance through major market crises.",
        body_style,
    ))
    elements.append(Spacer(1, 8))

    # --- Portfolio Weights ---
    elements.append(Paragraph("Portfolio Allocation", h2_style))
    w_data = [["Ticker", "Weight"]]
    for t, w in zip(tickers, weights):
        w_data.append([t, f"{w:.2%}"])
    elements.append(_make_table(w_data, col_widths=[2 * inch, 2 * inch]))
    elements.append(Spacer(1, 8))

    # --- Key Metrics ---
    elements.append(Paragraph("Key Risk Metrics", h2_style))
    m_data = [["Metric", "Value"]]
    for k, v in metrics.items():
        if isinstance(v, float):
            if abs(v) < 1 and k not in ("Sharpe Ratio", "Sortino Ratio", "Beta vs S&P500"):
                m_data.append([k, f"{v:.2%}"])
            else:
                m_data.append([k, f"{v:.4f}"])
        else:
            m_data.append([k, str(v)])
    elements.append(_make_table(m_data, col_widths=[3 * inch, 2.5 * inch]))
    elements.append(PageBreak())

    # --- Monte Carlo: GBM ---
    if gbm_paths is not None:
        elements.append(Paragraph("Monte Carlo Simulation — GBM", h2_style))
        fig = _plot_simulation_paths(gbm_paths, "GBM Simulation Paths (1 Year)")
        elements.append(_fig_to_image(fig))
        elements.append(Spacer(1, 4))
        fig2 = _plot_distribution(gbm_paths, "GBM — Distribution of Final Values")
        elements.append(_fig_to_image(fig2))

        if gbm_stats:
            elements.append(Spacer(1, 6))
            s_data = [["Statistic", "Value"]]
            for k, v in gbm_stats.items():
                if "Value" in k or "Percentile" in k or "VaR" in k or "CVaR" in k:
                    s_data.append([k, f"${v:,.0f}"])
                elif "Prob" in k:
                    s_data.append([k, f"{v:.2%}"])
                else:
                    s_data.append([k, f"{v:.4f}"])
            elements.append(_make_table(s_data, col_widths=[3 * inch, 2.5 * inch]))
        elements.append(PageBreak())

    # --- Monte Carlo: GARCH ---
    if garch_paths is not None:
        elements.append(Paragraph("Monte Carlo Simulation — GARCH(1,1)", h2_style))
        fig = _plot_simulation_paths(garch_paths, "GARCH(1,1) Simulation Paths (1 Year)")
        elements.append(_fig_to_image(fig))
        elements.append(Spacer(1, 4))
        fig2 = _plot_distribution(garch_paths, "GARCH — Distribution of Final Values")
        elements.append(_fig_to_image(fig2))

        if garch_stats:
            elements.append(Spacer(1, 6))
            s_data = [["Statistic", "Value"]]
            for k, v in garch_stats.items():
                if "Value" in k or "Percentile" in k or "VaR" in k or "CVaR" in k:
                    s_data.append([k, f"${v:,.0f}"])
                elif "Prob" in k:
                    s_data.append([k, f"{v:.2%}"])
                else:
                    s_data.append([k, f"{v:.4f}"])
            elements.append(_make_table(s_data, col_widths=[3 * inch, 2.5 * inch]))
        elements.append(PageBreak())

    # --- Stress Tests ---
    if stress_results:
        elements.append(Paragraph("Stress Test Results", h2_style))
        for r in stress_results:
            elements.append(Paragraph(f"<b>{r['name']}</b> ({r['start']} to {r['end']})", body_style))
            st_data = [
                ["Metric", "Value"],
                ["Cumulative Return", f"{r['cumulative_return']:.2%}"],
                ["Max Drawdown", f"{r['max_drawdown']:.2%}"],
                ["Annualised Volatility", f"{r['annualised_vol']:.2%}"],
                ["Worst Single Day", f"{r['worst_day']:.2%}"],
                ["Best Single Day", f"{r['best_day']:.2%}"],
            ]
            elements.append(_make_table(st_data, col_widths=[3 * inch, 2.5 * inch]))
            elements.append(Spacer(1, 8))
        elements.append(PageBreak())

    # --- Backtest ---
    if backtest_result is not None:
        elements.append(Paragraph("Backtest: Optimised vs Equal Weight", h2_style))
        fig = _plot_backtest(backtest_result)
        elements.append(_fig_to_image(fig))
        elements.append(Spacer(1, 8))

        # Comparison table
        opt_m = backtest_result["optimised"]["metrics"]
        ew_m = backtest_result["equal_weight"]["metrics"]
        bt_data = [["Metric", "Optimised", "Equal Weight"]]
        for k in opt_m:
            ov = opt_m[k]
            ev = ew_m[k]
            if isinstance(ov, float):
                if abs(ov) < 1 and k not in ("Sharpe Ratio", "Sortino Ratio", "Beta vs S&P500"):
                    bt_data.append([k, f"{ov:.2%}", f"{ev:.2%}"])
                else:
                    bt_data.append([k, f"{ov:.4f}", f"{ev:.4f}"])
            else:
                bt_data.append([k, str(ov), str(ev)])
        elements.append(_make_table(bt_data, col_widths=[2.5 * inch, 1.8 * inch, 1.8 * inch]))

    # --- Build PDF ---
    doc.build(elements)
    buf.seek(0)
    return buf.read()
