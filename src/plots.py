import numpy as np
import pandas as pd
import plotly.graph_objects as go


def efficient_frontier_fig(frontier, stock_stats, portfolios):
    """
    frontier: DataFrame with Return and Volatility columns (decimals)
    stock_stats: DataFrame, index = tickers, columns = Return, Volatility
    portfolios: dict of name -> (return, volatility)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
        mode="lines", name="Efficient frontier", line=dict(width=3),
    ))

    fig.add_trace(go.Scatter(
        x=stock_stats["Volatility"] * 100, y=stock_stats["Return"] * 100,
        mode="markers+text", text=[t.replace(".NS", "") for t in stock_stats.index],
        textposition="top center", name="Individual stocks",
        marker=dict(size=9, symbol="diamond"),
    ))

    for name, (ret, vol) in portfolios.items():
        fig.add_trace(go.Scatter(
            x=[vol * 100], y=[ret * 100], mode="markers", name=name,
            marker=dict(size=14, symbol="star"),
        ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annual volatility (%)",
        yaxis_title="Annual expected return (%)",
        height=500,
    )
    return fig


def growth_fig(results):
    """results: dict or DataFrame of name -> daily log returns."""
    fig = go.Figure()
    for name in results:
        daily = pd.Series(results[name]).fillna(0)
        fig.add_trace(go.Scatter(
            x=daily.index, y=np.exp(daily.cumsum()), mode="lines", name=name,
        ))
    fig.update_layout(
        title="Growth of Rs 1 invested (out-of-sample backtest)",
        xaxis_title="Date", yaxis_title="Portfolio value (Rs)", height=450,
    )
    return fig


def drawdown_fig(results):
    fig = go.Figure()
    for name in results:
        daily = pd.Series(results[name]).fillna(0)
        growth = np.exp(daily.cumsum())
        dd = (growth / growth.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=name))
    fig.update_layout(
        title="Drawdown (fall from previous peak)",
        xaxis_title="Date", yaxis_title="Drawdown (%)", height=450,
    )
    return fig


def weights_fig(weights, tickers, title="Portfolio weights"):
    labels = [t.replace(".NS", "") for t in tickers]
    fig = go.Figure(go.Bar(x=labels, y=np.asarray(weights) * 100))
    fig.update_layout(title=title, yaxis_title="Weight (%)", height=400)
    return fig


if __name__ == "__main__":
    import os
    from src.data_loader import load_saved_prices
    from src.returns import (compute_log_returns, annual_expected_returns,
                             annual_covariance, annual_volatility)
    from src.metrics import portfolio_return, portfolio_volatility
    from src.optimiser import (max_sharpe_portfolio, min_variance_portfolio,
                               efficient_frontier)

    prices = load_saved_prices()
    returns = compute_log_returns(prices)
    exp_ret = annual_expected_returns(returns)
    cov = annual_covariance(returns)

    w_min = min_variance_portfolio(cov)
    w_sharpe = max_sharpe_portfolio(exp_ret, cov)
    frontier = efficient_frontier(exp_ret, cov)

    stock_stats = pd.DataFrame({"Return": exp_ret, "Volatility": annual_volatility(returns)})
    portfolios = {
        "Min Variance": (portfolio_return(w_min, exp_ret), portfolio_volatility(w_min, cov)),
        "Max Sharpe": (portfolio_return(w_sharpe, exp_ret), portfolio_volatility(w_sharpe, cov)),
    }

    backtest_path = os.path.join("data", "processed", "backtest_returns.csv")
    results = pd.read_csv(backtest_path, index_col=0, parse_dates=True)

    figures = {
        "frontier": efficient_frontier_fig(frontier, stock_stats, portfolios),
        "growth": growth_fig(results),
        "drawdown": drawdown_fig(results),
        "weights": weights_fig(w_sharpe, returns.columns, "Max Sharpe weights"),
    }

    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    for name, fig in figures.items():
        fig.write_html(os.path.join("results", "figures", f"{name}.html"))
    print("Saved 4 charts to results/figures")

    figures["frontier"].show()