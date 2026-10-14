import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var_cvar(port_returns, confidence=0.95):
    """
    Historical method: look at what actually happened.
    VaR is the worst-(1-confidence) cutoff of past daily returns.
    """
    cutoff = np.percentile(port_returns, (1 - confidence) * 100)
    var = -cutoff
    cvar = -port_returns[port_returns <= cutoff].mean()
    return float(var), float(cvar)


def parametric_var_cvar(port_returns, confidence=0.95):
    """
    Parametric method: assume returns follow a normal (bell curve) distribution.
    Uses only the mean and standard deviation.
    """
    mu = port_returns.mean()
    sigma = port_returns.std()
    z = norm.ppf(1 - confidence)  # e.g. -1.645 for 95%
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    return float(var), float(cvar)


def monte_carlo_var_cvar(weights, returns, confidence=0.95, n_sims=100_000, seed=42):
    """
    Monte Carlo method: simulate many random days using the stocks'
    average returns and covariance, then measure the losses.
    """
    rng = np.random.default_rng(seed)
    mean = returns.mean().values
    cov = returns.cov().values

    simulated = rng.multivariate_normal(mean, cov, n_sims)  # n_sims x n_stocks
    port_sim = simulated @ weights                          # portfolio return per simulated day

    cutoff = np.percentile(port_sim, (1 - confidence) * 100)
    var = -cutoff
    cvar = -port_sim[port_sim <= cutoff].mean()
    return float(var), float(cvar)


def risk_table(weights, returns, confidence=0.95, portfolio_value=1_000_000):
    """VaR and CVaR from all three methods, in % and in rupees."""
    port_returns = returns.dot(weights)

    results = {
        "Historical": historical_var_cvar(port_returns, confidence),
        "Parametric": parametric_var_cvar(port_returns, confidence),
        "Monte Carlo": monte_carlo_var_cvar(weights, returns, confidence),
    }

    rows = []
    for method, (var, cvar) in results.items():
        rows.append({
            "Method": method,
            "VaR %": round(var * 100, 2),
            "CVaR %": round(cvar * 100, 2),
            "VaR (Rs)": round(var * portfolio_value),
            "CVaR (Rs)": round(cvar * portfolio_value),
        })
    return pd.DataFrame(rows).set_index("Method")


if __name__ == "__main__":
    from src.data_loader import load_saved_prices
    from src.returns import compute_log_returns, annual_expected_returns, annual_covariance
    from src.optimiser import max_sharpe_portfolio, min_variance_portfolio

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    prices = load_saved_prices()
    returns = compute_log_returns(prices)
    n = returns.shape[1]

    exp_ret = annual_expected_returns(returns)
    cov = annual_covariance(returns)

    portfolios = {
        "Equal Weight": np.array([1 / n] * n),
        "Min Variance": min_variance_portfolio(cov),
        "Max Sharpe": max_sharpe_portfolio(exp_ret, cov),
    }

    for name, w in portfolios.items():
        for conf in (0.95, 0.99):
            print(f"{name.upper()} | 1-day risk at {int(conf * 100)}% confidence "
                  f"(on Rs 10,00,000)")
            print(risk_table(w, returns, conf))
            print()