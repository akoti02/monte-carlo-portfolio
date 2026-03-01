# Monte Carlo Portfolio Risk Analysis Platform

A multi-page Streamlit web application for advanced portfolio risk analysis using Monte Carlo simulations, Markowitz optimisation, and historical stress testing.

## Features

- **Historical Data**: Pulls 5 years of data via yfinance for a configurable portfolio of 5–10 tickers
- **Monte Carlo Simulations**: 10,000 simulations using both Geometric Brownian Motion (GBM) and GARCH(1,1) volatility models, compared side by side
- **Risk Metrics**: VaR (95%/99%), CVaR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Beta vs S&P 500
- **Portfolio Optimisation**: Markowitz mean-variance optimisation with efficient frontier visualisation and optimal weight calculation via `scipy.optimize`
- **Stress Testing**: Replays portfolio through historical crisis periods (2008 GFC, 2020 COVID, 2022 Rate Hikes) using actual historical returns
- **Backtesting**: Compares the optimised portfolio against a naive equal-weight benchmark over the full historical period
- **PDF Reports**: Comprehensive downloadable report with charts, metrics tables, and methodology description (ReportLab)

## Pages

| Page | Description |
|------|-------------|
| **Overview & Metrics** | Configure tickers, fetch data, view price history, correlations, and key risk/return metrics |
| **Monte Carlo Simulator** | Run GBM and GARCH(1,1) simulations, compare path distributions and statistics |
| **Portfolio Optimiser** | Efficient frontier, max Sharpe / min variance portfolios, weight comparison, backtest preview |
| **Stress Test Results** | Historical crisis replay with per-crisis charts and impact comparison |
| **Report Download** | Generate and download a comprehensive PDF report |

## Setup

```bash
cd portfolio_monte_carlo
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project Structure

```
portfolio_monte_carlo/
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── data.py                     # yfinance data fetching
│   ├── metrics.py                  # Risk/return metric calculations
│   ├── monte_carlo.py              # GBM and GARCH(1,1) simulation engines
│   ├── optimizer.py                # Markowitz mean-variance optimisation
│   ├── stress_test.py              # Historical crisis replay
│   ├── backtest.py                 # Backtesting engine
│   └── report.py                   # PDF report generation (ReportLab)
└── pages/
    ├── 1_Overview_Metrics.py
    ├── 2_Monte_Carlo_Simulator.py
    ├── 3_Portfolio_Optimiser.py
    ├── 4_Stress_Test.py
    └── 5_Report_Download.py
```

## Methodology

### Monte Carlo — GBM
Assumes constant drift (μ) and volatility (σ) estimated from historical returns:

```
S(t) = S(0) × exp((μ - σ²/2)t + σW(t))
```

### Monte Carlo — GARCH(1,1)
Models time-varying conditional variance using the `arch` library:

```
σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
```

This captures volatility clustering observed in real markets, producing heavier tails than GBM.

### Portfolio Optimisation
Uses `scipy.optimize.minimize` with SLSQP to solve the Markowitz problem:
- **Max Sharpe**: Maximises (return − risk-free rate) / volatility
- **Min Variance**: Minimises portfolio volatility
- Constraints: weights sum to 1, long-only (0 ≤ w ≤ 1)

### Stress Testing
Replays actual historical returns during defined crisis windows rather than simulating hypothetical scenarios.
