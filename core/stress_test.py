"""Stress testing module — replay portfolio through historical crises."""

import numpy as np
import pandas as pd
from core.metrics import max_drawdown, annualised_volatility

# Historical crisis periods (start, end, label)
CRISIS_PERIODS = {
    "2008 GFC": ("2007-10-01", "2009-03-31"),
    "2020 COVID Crash": ("2020-02-01", "2020-04-30"),
    "2022 Rate Hikes": ("2022-01-01", "2022-10-31"),
}


def replay_crisis(
    returns: pd.DataFrame,
    weights: np.ndarray,
    crisis_name: str,
    start: str,
    end: str,
) -> dict | None:
    """Replay portfolio performance during a specific crisis period.

    Returns None if the data doesn't cover the crisis period.
    """
    mask = (returns.index >= start) & (returns.index <= end)
    crisis_returns = returns.loc[mask]

    if len(crisis_returns) < 5:
        return None

    port_returns = (crisis_returns * weights).sum(axis=1)
    cum_returns = (1 + port_returns).cumprod()

    return {
        "name": crisis_name,
        "start": start,
        "end": end,
        "trading_days": len(port_returns),
        "cumulative_return": float(cum_returns.iloc[-1] - 1),
        "max_drawdown": float(max_drawdown(port_returns)),
        "annualised_vol": float(annualised_volatility(port_returns)),
        "worst_day": float(port_returns.min()),
        "best_day": float(port_returns.max()),
        "daily_returns": port_returns,
        "cumulative": cum_returns,
    }


def run_all_stress_tests(
    returns: pd.DataFrame, weights: np.ndarray
) -> list[dict]:
    """Run stress tests for all defined crisis periods.

    Only returns results for periods covered by the data.
    """
    results = []
    for name, (start, end) in CRISIS_PERIODS.items():
        result = replay_crisis(returns, weights, name, start, end)
        if result is not None:
            results.append(result)
    return results


def stress_test_summary_df(results: list[dict]) -> pd.DataFrame:
    """Create a summary DataFrame from stress test results."""
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            "Crisis": r["name"],
            "Period": f"{r['start']} to {r['end']}",
            "Days": r["trading_days"],
            "Cumulative Return": f"{r['cumulative_return']:.2%}",
            "Max Drawdown": f"{r['max_drawdown']:.2%}",
            "Annualised Vol": f"{r['annualised_vol']:.2%}",
            "Worst Day": f"{r['worst_day']:.2%}",
            "Best Day": f"{r['best_day']:.2%}",
        })
    return pd.DataFrame(rows)
