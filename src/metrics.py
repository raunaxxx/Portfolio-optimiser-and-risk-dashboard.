import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06  # assumed 6% per year, roughly an Indian government bond yield


def portfolio_return(weights, exp_returns):
    """Expected annual return of a portfolio: sum of weight x stock return."""
    return float(np.dot(weights, exp_returns))


def portfolio_volatility(weights, cov_matrix):
    """Annual volatility of a portfolio: sqrt(w' * Cov * w)."""
    return float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))


def sharpe_ratio(ret, vol, rf=RISK_FREE_RATE):
    """Extra return earned per unit of risk taken."""
    return (ret - rf) / vol


def portfolio_daily_returns(weights, returns):
    """Daily return series of a portfolio with fixed weights."""
    return returns.dot(weights)


def cumulative_returns(daily_returns):
    """Growth of 1 rupee invested: e.g. 1.8 means +80%."""
    return np.exp(daily_returns.cumsum())


def max_drawdown(daily_returns):
    """Largest peak-to-trough fall in portfolio value (a negative number)."""
    growth = cumulative_returns(daily_returns)
    peak = growth.cummax()
    drawdown = growth / peak - 1
    return float(drawdown.min())


def sortino_ratio(daily_returns, rf=RISK_FREE_RATE):
    """Like Sharpe, but only counts downside volatility as risk."""
    ann_ret = daily_returns.mean() * TRADING_DAYS
    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    return (ann_ret - rf) / downside_vol


def summary(daily_returns, rf=RISK_FREE_RATE):
    """All key metrics for a daily return series, as a dictionary."""
    ann_ret = daily_returns.mean() * TRADING_DAYS
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    return {
        "Annual Return": ann_ret,
        "Volatility": ann_vol,
        "Sharpe": sharpe_ratio(ann_ret, ann_vol, rf),
        "Sortino": sortino_ratio(daily_returns, rf),
        "Max Drawdown": max_drawdown(daily_returns),
    }


if __name__ == "__main__":
    from src.data_loader import load_saved_prices
    from src.returns import compute_log_returns, annual_expected_returns, annual_covariance

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    prices = load_saved_prices()
    returns = compute_log_returns(prices)
    n = returns.shape[1]

    # Test with an equal-weight portfolio (20% in each stock)
    weights = np.array([1 / n] * n)

    exp_ret = annual_expected_returns(returns)
    cov = annual_covariance(returns)

    ret = portfolio_return(weights, exp_ret)
    vol = portfolio_volatility(weights, cov)
    print(f"Equal-weight return:     {ret * 100:.2f}%")
    print(f"Equal-weight volatility: {vol * 100:.2f}%")
    print(f"Equal-weight Sharpe:     {sharpe_ratio(ret, vol):.2f}")

    daily = portfolio_daily_returns(weights, returns)
    print("\nFull summary:")
    for name, value in summary(daily).items():
        print(f"  {name}: {value:.3f}")