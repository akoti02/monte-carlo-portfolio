"""Backtesting engine — compare optimised portfolio vs equal-weight benchmark."""

import numpy as np
import pandas as pd
from core.metrics import (
    compute_all_metrics,
    portfolio_returns,
    annualised_return,
    annualised_volatility,
    max_drawdown,
)


def backtest_portfolio(
    returns: pd.DataFrame,
    optimised_weights: np.ndarray,
    benchmark_returns: pd.Series | None = None,
    initial_value: float = 100_000,
) -> dict:
    """Backtest optimised weights vs equal-weight over the historical period."""
    n_assets = returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets

    opt_daily = portfolio_returns(returns, optimised_weights)
    ew_daily = portfolio_returns(returns, equal_weights)

    opt_cum = initial_value * (1 + opt_daily).cumprod()
    ew_cum = initial_value * (1 + ew_daily).cumprod()

    opt_metrics = compute_all_metrics(opt_daily, benchmark_returns)
    ew_metrics = compute_all_metrics(ew_daily, benchmark_returns)

    # Rolling metrics (252-day window)
    window = min(252, len(returns) // 2)
    opt_rolling_vol = opt_daily.rolling(window).std() * np.sqrt(252)
    ew_rolling_vol = ew_daily.rolling(window).std() * np.sqrt(252)

    opt_rolling_sharpe = (
        opt_daily.rolling(window).mean() * 252 - 0.04
    ) / (opt_daily.rolling(window).std() * np.sqrt(252))
    ew_rolling_sharpe = (
        ew_daily.rolling(window).mean() * 252 - 0.04
    ) / (ew_daily.rolling(window).std() * np.sqrt(252))

    return {
        "optimised": {
            "daily_returns": opt_daily,
            "cumulative": opt_cum,
            "metrics": opt_metrics,
            "rolling_vol": opt_rolling_vol,
            "rolling_sharpe": opt_rolling_sharpe,
            "weights": optimised_weights,
        },
        "equal_weight": {
            "daily_returns": ew_daily,
            "cumulative": ew_cum,
            "metrics": ew_metrics,
            "rolling_vol": ew_rolling_vol,
            "rolling_sharpe": ew_rolling_sharpe,
            "weights": equal_weights,
        },
    }


def comparison_table(backtest_result: dict) -> pd.DataFrame:
    """Create a side-by-side metrics comparison table."""
    opt_m = backtest_result["optimised"]["metrics"]
    ew_m = backtest_result["equal_weight"]["metrics"]

    rows = []
    for key in opt_m:
        rows.append({
            "Metric": key,
            "Optimised": opt_m[key],
            "Equal Weight": ew_m[key],
        })

    df = pd.DataFrame(rows).set_index("Metric")

    # Format percentages
    pct_metrics = [
        "Annualised Return", "Annualised Volatility", "Max Drawdown",
        "VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%",
    ]
    for col in ["Optimised", "Equal Weight"]:
        for m in pct_metrics:
            if m in df.index:
                df.loc[m, col] = f"{df.loc[m, col]:.2%}"
        for m in ["Sharpe Ratio", "Sortino Ratio", "Beta vs S&P500"]:
            if m in df.index:
                df.loc[m, col] = f"{df.loc[m, col]:.4f}"

    return df
