"""Monte Carlo simulation engines: Geometric Brownian Motion and GARCH(1,1)."""

import numpy as np
import pandas as pd
from arch import arch_model


TRADING_DAYS = 252


def simulate_gbm(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    n_days: int = 252,
    initial_value: float = 100_000,
) -> np.ndarray:
    """Geometric Brownian Motion simulation.

    Returns array of shape (n_simulations, n_days) of portfolio values.
    """
    port_returns = (returns * weights).sum(axis=1)
    mu = port_returns.mean()
    sigma = port_returns.std()

    # Generate random returns
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n_simulations, n_days))

    # GBM: S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
    daily_returns = (mu - 0.5 * sigma**2) + sigma * Z
    cum_returns = np.cumsum(daily_returns, axis=1)
    paths = initial_value * np.exp(cum_returns)

    return paths


def fit_garch_and_simulate(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10_000,
    n_days: int = 252,
    initial_value: float = 100_000,
) -> np.ndarray:
    """GARCH(1,1) simulation with time-varying volatility.

    Returns array of shape (n_simulations, n_days) of portfolio values.
    """
    port_returns = (returns * weights).sum(axis=1)
    scaled = port_returns * 100  # arch library works better with percentage returns

    # Fit GARCH(1,1)
    model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    result = model.fit(disp="off", show_warning=False)

    mu = result.params.get("mu", 0.0)
    omega = result.params.get("omega", 0.01)
    alpha = result.params.get("alpha[1]", 0.05)
    beta = result.params.get("beta[1]", 0.90)

    # Last conditional variance from the fit
    last_var = result.conditional_volatility.iloc[-1] ** 2

    rng = np.random.default_rng(42)
    paths = np.zeros((n_simulations, n_days))

    for i in range(n_simulations):
        h_t = last_var
        log_return_sum = 0.0
        for t in range(n_days):
            z = rng.standard_normal()
            ret = (mu + np.sqrt(h_t) * z) / 100  # back to decimal
            log_return_sum += ret
            paths[i, t] = initial_value * np.exp(log_return_sum)
            # Update variance: GARCH(1,1) recursion
            epsilon = np.sqrt(h_t) * z
            h_t = omega + alpha * epsilon**2 + beta * h_t

    return paths


def simulation_statistics(paths: np.ndarray, confidence_levels=(0.95, 0.99)) -> dict:
    """Compute summary statistics from simulation paths."""
    final_values = paths[:, -1]
    initial = paths[0, 0] if paths.shape[1] > 0 else 100_000
    total_returns = (final_values - initial) / initial

    stats = {
        "Mean Final Value": np.mean(final_values),
        "Median Final Value": np.median(final_values),
        "Std Final Value": np.std(final_values),
        "Mean Return": np.mean(total_returns),
        "5th Percentile": np.percentile(final_values, 5),
        "95th Percentile": np.percentile(final_values, 95),
        "Prob of Loss": np.mean(final_values < initial),
    }

    for cl in confidence_levels:
        pct = (1 - cl) * 100
        var_val = initial - np.percentile(final_values, pct)
        stats[f"VaR {int(cl*100)}%"] = var_val
        # CVaR: average loss beyond VaR
        threshold = np.percentile(final_values, pct)
        tail = final_values[final_values <= threshold]
        stats[f"CVaR {int(cl*100)}%"] = initial - np.mean(tail) if len(tail) > 0 else var_val

    return stats
