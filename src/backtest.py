import os
import numpy as np
import pandas as pd

from src.metrics import summary, cumulative_returns
from src.optimiser import max_sharpe_portfolio, min_variance_portfolio
from src.returns import annual_expected_returns, annual_covariance

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def rolling_backtest(returns, strategy, window=252, rebalance_every=21,
                     cost=0.001, max_weight=0.4):
    """
    Walk-forward backtest.
    strategy: "max_sharpe", "min_variance" or "equal"
    cost: transaction cost per rupee traded (0.001 = 0.1%)
    Returns a Series of out-of-sample daily portfolio returns.
    """
    n = returns.shape[1]
    prev_w = np.array([1 / n] * n)
    pieces = []

    for start in range(window, len(returns), rebalance_every):
        train = returns.iloc[start - window:start]               # past data only
        test = returns.iloc[start:start + rebalance_every]       # unseen future

        if strategy == "max_sharpe":
            w = max_sharpe_portfolio(annual_expected_returns(train),
                                     annual_covariance(train),
                                     max_weight=max_weight)
        elif strategy == "min_variance":
            w = min_variance_portfolio(annual_covariance(train),
                                       max_weight=max_weight)
        else:
            w = np.array([1 / n] * n)

        port = test.dot(w)

        # Pay transaction costs on the amount we changed our weights by
        turnover = np.abs(w - prev_w).sum()
        port.iloc[0] -= cost * turnover
        prev_w = w

        pieces.append(port)

    return pd.concat(pieces)


def benchmark_returns(index):
    """Nifty 50 daily log returns, lined up with our backtest dates."""
    from src.data_loader import BENCHMARK, download_prices

    px = download_prices(BENCHMARK, save=False).iloc[:, 0]
    px = px.reindex(index.union(px.index)).ffill()
    log_ret = np.log(px / px.shift(1))
    return log_ret.reindex(index)


def backtest_table(results):
    """Summary metrics for each strategy, formatted for printing."""
    rows = {}
    for name, daily in results.items():
        s = summary(daily)
        rows[name] = {
            "Annual Return %": round(s["Annual Return"] * 100, 2),
            "Volatility %": round(s["Volatility"] * 100, 2),
            "Sharpe": round(s["Sharpe"], 2),
            "Sortino": round(s["Sortino"], 2),
            "Max Drawdown %": round(s["Max Drawdown"] * 100, 2),
            "Rs 1 becomes": round(float(cumulative_returns(daily).iloc[-1]), 2),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from src.data_loader import load_saved_prices
    from src.returns import compute_log_returns

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    prices = load_saved_prices()
    returns = compute_log_returns(prices)

    results = {
        "Max Sharpe": rolling_backtest(returns, "max_sharpe"),
        "Min Variance": rolling_backtest(returns, "min_variance"),
        "Equal Weight": rolling_backtest(returns, "equal"),
    }
    results["Nifty 50"] = benchmark_returns(results["Equal Weight"].index)

    print(f"Out-of-sample period: {results['Equal Weight'].index[0].date()} "
          f"to {results['Equal Weight'].index[-1].date()}\n")
    print(backtest_table(results))

    # Save for the dashboard later
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(PROCESSED_DIR, "backtest_returns.csv"))