"""Risk and performance metrics calculations."""

import numpy as np
import pandas as pd


TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # annualised


def portfolio_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """Compute weighted portfolio daily returns."""
    return (returns * weights).sum(axis=1)


def annualised_return(daily_returns: pd.Series) -> float:
    return daily_returns.mean() * TRADING_DAYS


def annualised_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    ann_ret = annualised_return(daily_returns)
    ann_vol = annualised_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - rf) / ann_vol


def sortino_ratio(daily_returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    ann_ret = annualised_return(daily_returns)
    downside = daily_returns[daily_returns < 0].std() * np.sqrt(TRADING_DAYS)
    if downside == 0:
        return 0.0
    return (ann_ret - rf) / downside


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown from cumulative returns series."""
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    return drawdowns.min()


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (negative number = loss)."""
    if len(daily_returns) == 0:
        return 0.0
    return float(np.percentile(daily_returns, (1 - confidence) * 100))


def conditional_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR."""
    if len(daily_returns) == 0:
        return 0.0
    var = value_at_risk(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def beta_vs_benchmark(
    portfolio_returns_s: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Beta of portfolio against benchmark."""
    aligned = pd.DataFrame(
        {"port": portfolio_returns_s, "bench": benchmark_returns}
    ).dropna()
    if len(aligned) < 2:
        return 0.0
    cov = np.cov(aligned["port"], aligned["bench"])
    if cov[1, 1] == 0:
        return 0.0
    return cov[0, 1] / cov[1, 1]


def compute_all_metrics(
    daily_returns: pd.Series, benchmark_returns: pd.Series | None = None
) -> dict:
    """Compute a full suite of risk/return metrics."""
    metrics = {
        "Annualised Return": annualised_return(daily_returns),
        "Annualised Volatility": annualised_volatility(daily_returns),
        "Sharpe Ratio": sharpe_ratio(daily_returns),
        "Sortino Ratio": sortino_ratio(daily_returns),
        "Max Drawdown": max_drawdown(daily_returns),
        "VaR 95%": value_at_risk(daily_returns, 0.95),
        "VaR 99%": value_at_risk(daily_returns, 0.99),
        "CVaR 95%": conditional_var(daily_returns, 0.95),
        "CVaR 99%": conditional_var(daily_returns, 0.99),
    }
    if benchmark_returns is not None:
        metrics["Beta vs S&P500"] = beta_vs_benchmark(
            daily_returns, benchmark_returns
        )
    return metrics
