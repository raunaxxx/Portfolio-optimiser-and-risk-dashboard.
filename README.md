## Key Results

**In-sample (2019-2024, fitted and scored on the same data)**
| Portfolio | Ann. Return | Volatility | Sharpe |
|-----------|-------------|------------|--------|
| Max Sharpe | 17.57% | 20.74% | 0.56 |
| Min Variance | 13.80% | 17.69% | 0.44 |
| Equal Weight | 14.68% | 18.19% | 0.48 |

**Out-of-sample walk-forward backtest (Jan 2020 - Dec 2024)**
252-day training window, monthly rebalance, 0.1% transaction cost, 40% max weight.
| Strategy | Ann. Return | Volatility | Sharpe | Max Drawdown |
|----------|-------------|------------|--------|--------------|
| Max Sharpe | 16.38% | 21.24% | 0.49 | -37.76% |
| Equal Weight | 14.63% | 19.13% | 0.45 | -35.34% |
| Min Variance | 13.65% | 19.04% | 0.40 | -36.60% |
| Nifty 50 (price index) | 13.31% | 19.31% | 0.38 | -38.41% |

**Findings**
- Max Sharpe's in-sample Sharpe (0.56) overstated its out-of-sample Sharpe (0.49),
  which shows the overfitting problem of mean-variance optimisation.
- Equal weight beat minimum variance after costs.
- Historical 99% CVaR was about 1.7x higher than parametric and Monte Carlo
  estimates, because the normal distribution underestimates extreme days like March 2020.