import numpy as np
import pandas as pd

TRADING_DAYS = 252  # trading days in a year


def compute_log_returns(prices):
    """
    Daily log returns: ln(today's price / yesterday's price).
    Log returns can be added across days, which makes the maths cleaner.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def annual_expected_returns(returns):
    """Average daily return, scaled up to a yearly figure."""
    return returns.mean() * TRADING_DAYS


def annual_covariance(returns):
    """
    Covariance matrix of returns, scaled to a year.
    It shows how much each pair of stocks moves together.
    """
    return returns.cov() * TRADING_DAYS


def annual_volatility(returns):
    """Standard deviation of returns, scaled to a year."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def correlation_matrix(returns):
    """Correlation between stocks, from -1 to +1."""
    return returns.corr()


if __name__ == "__main__":
    from src.data_loader import load_saved_prices

    prices = load_saved_prices()
    returns = compute_log_returns(prices)

    print("Annual expected returns:")
    print((annual_expected_returns(returns) * 100).round(2))

    print("\nAnnual volatility:")
    print((annual_volatility(returns) * 100).round(2))

    print("\nCorrelation matrix:")
    print(correlation_matrix(returns).round(2))