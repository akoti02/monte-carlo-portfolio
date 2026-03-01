"""Markowitz mean-variance portfolio optimisation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


def portfolio_performance(
    weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray
) -> tuple[float, float]:
    """Annualised return and volatility for given weights."""
    ret = np.dot(weights, mean_returns) * TRADING_DAYS
    vol = np.sqrt(np.dot(weights, np.dot(cov_matrix * TRADING_DAYS, weights)))
    return ret, vol


def negative_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = RISK_FREE_RATE,
) -> float:
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if vol == 0:
        return 0.0
    return -(ret - rf) / vol


def minimize_volatility(
    weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray
) -> float:
    _, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return vol


def optimise_max_sharpe(
    returns: pd.DataFrame, rf: float = RISK_FREE_RATE
) -> dict:
    """Find the maximum Sharpe ratio portfolio."""
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.ones(n) / n

    result = minimize(
        negative_sharpe,
        initial,
        args=(mean_ret, cov, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    ret, vol = portfolio_performance(result.x, mean_ret, cov)
    return {
        "weights": result.x,
        "return": ret,
        "volatility": vol,
        "sharpe": (ret - rf) / vol if vol > 0 else 0,
    }


def optimise_min_variance(returns: pd.DataFrame) -> dict:
    """Find the minimum variance portfolio."""
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.ones(n) / n

    result = minimize(
        minimize_volatility,
        initial,
        args=(mean_ret, cov),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    ret, vol = portfolio_performance(result.x, mean_ret, cov)
    return {
        "weights": result.x,
        "return": ret,
        "volatility": vol,
        "sharpe": (ret - RISK_FREE_RATE) / vol if vol > 0 else 0,
    }


def compute_efficient_frontier(
    returns: pd.DataFrame, n_points: int = 100
) -> dict:
    """Compute the efficient frontier.

    Returns dict with arrays: returns, volatilities, sharpe_ratios, and
    the list of weight arrays.
    """
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values

    # Find min and max feasible returns
    min_var = optimise_min_variance(returns)
    max_sharpe = optimise_max_sharpe(returns)

    # Individual asset max return
    max_ret = max(mean_ret) * TRADING_DAYS
    target_returns = np.linspace(min_var["return"], max_ret, n_points)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, t=target: (
                    np.dot(w, mean_ret) * TRADING_DAYS - t
                ),
            },
        ]
        bounds = tuple((0, 1) for _ in range(n))
        initial = np.ones(n) / n

        result = minimize(
            minimize_volatility,
            initial,
            args=(mean_ret, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            ret, vol = portfolio_performance(result.x, mean_ret, cov)
            frontier_rets.append(ret)
            frontier_vols.append(vol)
            frontier_weights.append(result.x)

    frontier_rets = np.array(frontier_rets)
    frontier_vols = np.array(frontier_vols)
    sharpes = np.where(
        frontier_vols > 0,
        (frontier_rets - RISK_FREE_RATE) / frontier_vols,
        0,
    )

    return {
        "returns": frontier_rets,
        "volatilities": frontier_vols,
        "sharpe_ratios": sharpes,
        "weights": frontier_weights,
        "max_sharpe": max_sharpe,
        "min_variance": min_var,
    }


def generate_random_portfolios(
    returns: pd.DataFrame, n_portfolios: int = 5000
) -> dict:
    """Generate random portfolios for plotting."""
    n = returns.shape[1]
    mean_ret = returns.mean().values
    cov = returns.cov().values
    rng = np.random.default_rng(42)

    all_rets = []
    all_vols = []
    all_sharpes = []

    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        ret, vol = portfolio_performance(w, mean_ret, cov)
        all_rets.append(ret)
        all_vols.append(vol)
        sr = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        all_sharpes.append(sr)

    return {
        "returns": np.array(all_rets),
        "volatilities": np.array(all_vols),
        "sharpe_ratios": np.array(all_sharpes),
    }
