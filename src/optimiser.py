import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.metrics import (
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
    RISK_FREE_RATE,
)


def _setup(n, max_weight):
    """Rules for the optimiser: weights add up to 100%, none below 0 (no short selling)."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0, max_weight) for _ in range(n))
    x0 = np.array([1 / n] * n)  # start from equal weights
    return constraints, bounds, x0


def min_variance_portfolio(cov, max_weight=1.0):
    """Weights that give the lowest possible volatility."""
    cov = np.asarray(cov)
    n = cov.shape[0]
    constraints, bounds, x0 = _setup(n, max_weight)

    result = minimize(
        lambda w: portfolio_volatility(w, cov),
        x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x


def max_sharpe_portfolio(exp_ret, cov, rf=RISK_FREE_RATE, max_weight=1.0):
    """Weights that give the highest Sharpe ratio."""
    exp_ret = np.asarray(exp_ret)
    cov = np.asarray(cov)
    n = len(exp_ret)
    constraints, bounds, x0 = _setup(n, max_weight)

    def negative_sharpe(w):
        ret = portfolio_return(w, exp_ret)
        vol = portfolio_volatility(w, cov)
        return -sharpe_ratio(ret, vol, rf)  # minimise the negative = maximise Sharpe

    result = minimize(
        negative_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x


def efficient_frontier(exp_ret, cov, n_points=50, rf=RISK_FREE_RATE, max_weight=1.0):
    """
    For a range of target returns, find the lowest-risk portfolio that achieves each one.
    Returns a DataFrame with return, volatility and Sharpe for each point.
    """
    exp_ret = np.asarray(exp_ret)
    cov = np.asarray(cov)
    n = len(exp_ret)
    _, bounds, x0 = _setup(n, max_weight)

    # Targets run from the min-variance portfolio's return up to the best single stock
    w_min = min_variance_portfolio(cov, max_weight)
    low = portfolio_return(w_min, exp_ret)
    high = exp_ret.max()

    rows = []
    for target in np.linspace(low, high, n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, exp_ret) - t},
        ]
        result = minimize(
            lambda w: portfolio_volatility(w, cov),
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
        )
        if result.success:
            vol = portfolio_volatility(result.x, cov)
            rows.append({
                "Return": target,
                "Volatility": vol,
                "Sharpe": sharpe_ratio(target, vol, rf),
            })

    return pd.DataFrame(rows)


def weights_table(weights, tickers):
    """Turn a weights array into a labelled table, in percent."""
    return pd.Series(weights, index=tickers).mul(100).round(2)


if __name__ == "__main__":
    from src.data_loader import load_saved_prices
    from src.returns import compute_log_returns, annual_expected_returns, annual_covariance

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    prices = load_saved_prices()
    returns = compute_log_returns(prices)
    tickers = returns.columns

    exp_ret = annual_expected_returns(returns)
    cov = annual_covariance(returns)

    # 1. Minimum variance portfolio
    w_min = min_variance_portfolio(cov)
    r, v = portfolio_return(w_min, exp_ret), portfolio_volatility(w_min, cov)
    print("MIN VARIANCE PORTFOLIO")
    print(weights_table(w_min, tickers))
    print(f"Return {r*100:.2f}% | Volatility {v*100:.2f}% | Sharpe {sharpe_ratio(r, v):.2f}\n")

    # 2. Maximum Sharpe portfolio
    w_sharpe = max_sharpe_portfolio(exp_ret, cov)
    r, v = portfolio_return(w_sharpe, exp_ret), portfolio_volatility(w_sharpe, cov)
    print("MAX SHARPE PORTFOLIO")
    print(weights_table(w_sharpe, tickers))
    print(f"Return {r*100:.2f}% | Volatility {v*100:.2f}% | Sharpe {sharpe_ratio(r, v):.2f}\n")

    # 3. Efficient frontier
    frontier = efficient_frontier(exp_ret, cov)
    print("EFFICIENT FRONTIER (first 3 and last 3 points)")
    print((frontier.head(3) * 100).round(2))
    print((frontier.tail(3) * 100).round(2))