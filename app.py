import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import DEFAULT_TICKERS, BENCHMARK, download_prices
from src.returns import (compute_log_returns, annual_expected_returns,
                         annual_covariance, annual_volatility)
from src.metrics import (portfolio_return, portfolio_volatility, sharpe_ratio,
                         portfolio_daily_returns, summary, RISK_FREE_RATE)
from src.optimiser import (min_variance_portfolio, max_sharpe_portfolio,
                           efficient_frontier)
from src.risk import risk_table
from src.backtest import rolling_backtest, backtest_table
from src.plots import (efficient_frontier_fig, growth_fig, drawdown_fig,
                       weights_fig)

st.set_page_config(page_title="Portfolio Optimiser", layout="wide")

STOCK_OPTIONS = DEFAULT_TICKERS + [
    "SBIN.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "LT.NS",
    "BHARTIARTL.NS", "MARUTI.NS", "SUNPHARMA.NS", "ASIANPAINT.NS",
]


# ---------- cached helpers (so the app doesn't re-download on every click) ----------
@st.cache_data(show_spinner="Downloading prices...")
def get_prices(tickers, start, end):
    return download_prices(list(tickers), start=str(start), end=str(end), save=False)


@st.cache_data(show_spinner="Running walk-forward backtest (this can take a minute)...")
def run_backtests(returns, window, rebalance, cost, max_weight, start, end):
    results = {
        "Max Sharpe": rolling_backtest(returns, "max_sharpe", window, rebalance, cost, max_weight),
        "Min Variance": rolling_backtest(returns, "min_variance", window, rebalance, cost, max_weight),
        "Equal Weight": rolling_backtest(returns, "equal", window, rebalance, cost, max_weight),
    }
    idx = results["Equal Weight"].index

    bench = download_prices(BENCHMARK, start=str(start), end=str(end), save=False).iloc[:, 0]
    bench = bench.reindex(idx.union(bench.index)).ffill()
    bench_ret = np.log(bench / bench.shift(1)).reindex(idx).fillna(0)
    results["Nifty 50"] = bench_ret

    return pd.DataFrame(results)


# ---------- sidebar ----------
st.sidebar.title("Settings")
tickers = st.sidebar.multiselect("Stocks (NSE)", STOCK_OPTIONS, default=DEFAULT_TICKERS)
start = st.sidebar.date_input("Start date", dt.date(2019, 1, 1))
end = st.sidebar.date_input("End date", dt.date(2025, 1, 1))
max_w = st.sidebar.slider("Max weight per stock", 0.2, 1.0, 0.4, 0.05)
confidence = st.sidebar.selectbox("VaR confidence", [0.95, 0.99], index=0)
value = st.sidebar.number_input("Portfolio value (Rs)", 10_000, 100_000_000, 1_000_000, 10_000)

with st.sidebar.expander("Backtest settings"):
    window = st.number_input("Training window (trading days)", 60, 756, 252, 21)
    rebalance = st.number_input("Rebalance every (trading days)", 5, 126, 21, 1)
    cost_pct = st.number_input("Transaction cost (%)", 0.0, 1.0, 0.1, 0.05)

st.title("Portfolio Optimiser & Risk Dashboard")
st.caption(f"Mean-variance optimisation, VaR/CVaR and walk-forward backtest. "
           f"Assumed risk-free rate: {RISK_FREE_RATE * 100:.0f}% per year.")

# ---------- checks ----------
if len(tickers) < 2:
    st.warning("Pick at least 2 stocks in the sidebar.")
    st.stop()
if max_w * len(tickers) < 1:
    st.error(f"Max weight too low: with {len(tickers)} stocks it must be at least "
             f"{1 / len(tickers):.2f}.")
    st.stop()

# ---------- data and optimisation ----------
prices = get_prices(tuple(tickers), start, end)
returns = compute_log_returns(prices)
n = returns.shape[1]

exp_ret = annual_expected_returns(returns)
cov = annual_covariance(returns)

portfolios = {
    "Equal Weight": np.array([1 / n] * n),
    "Min Variance": min_variance_portfolio(cov, max_w),
    "Max Sharpe": max_sharpe_portfolio(exp_ret, cov, max_weight=max_w),
}

tab1, tab2, tab3 = st.tabs(["Optimisation", "Risk (VaR / CVaR)", "Backtest"])

# ---------- tab 1: optimisation ----------
with tab1:
    st.subheader("Efficient frontier")
    frontier = efficient_frontier(exp_ret, cov, max_weight=max_w)
    stock_stats = pd.DataFrame({"Return": exp_ret, "Volatility": annual_volatility(returns)})
    points = {name: (portfolio_return(w, exp_ret), portfolio_volatility(w, cov))
              for name, w in portfolios.items() if name != "Equal Weight"}
    st.plotly_chart(efficient_frontier_fig(frontier, stock_stats, points),
                    use_container_width=True)

    st.subheader("Portfolio comparison (in-sample)")
    rows = {}
    for name, w in portfolios.items():
        r, v = portfolio_return(w, exp_ret), portfolio_volatility(w, cov)
        rows[name] = {"Return %": round(r * 100, 2), "Volatility %": round(v * 100, 2),
                      "Sharpe": round(sharpe_ratio(r, v), 2)}
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
    st.caption("In-sample: weights are fitted on the same data they are scored on, "
               "so these numbers look better than real performance. See the Backtest tab.")

    st.subheader("Weights")
    weights_df = pd.DataFrame(portfolios, index=returns.columns).mul(100).round(2)
    c1, c2 = st.columns([1, 1])
    c1.dataframe(weights_df, use_container_width=True)
    choice = c2.selectbox("Show chart for", list(portfolios.keys()), index=2)
    c2.plotly_chart(weights_fig(portfolios[choice], returns.columns, f"{choice} weights"),
                    use_container_width=True)

# ---------- tab 2: risk ----------
with tab2:
    st.subheader(f"1-day risk at {int(confidence * 100)}% confidence (Rs {value:,.0f})")
    for name, w in portfolios.items():
        st.markdown(f"**{name}**")
        st.dataframe(risk_table(w, returns, confidence, value), use_container_width=True)
    st.caption("Historical VaR/CVaR uses real past losses. Parametric and Monte Carlo "
               "assume normal returns, so they usually understate extreme losses.")

# ---------- tab 3: backtest ----------
with tab3:
    st.subheader("Walk-forward backtest (out-of-sample)")
    st.write(f"Each rebalance uses the previous {window} trading days to choose weights, "
             f"then holds them for {rebalance} days on unseen data.")

    if len(returns) < window + rebalance:
        st.warning("Not enough data for this training window. Widen the date range.")
    else:
        if st.button("Run backtest"):
            st.session_state["bt"] = run_backtests(
                returns, int(window), int(rebalance), cost_pct / 100, max_w, start, end)

        bt = st.session_state.get("bt")
        if bt is None:
            st.info("Click **Run backtest** to start.")
        else:
            st.dataframe(backtest_table(bt), use_container_width=True)
            st.plotly_chart(growth_fig(bt), use_container_width=True)
            st.plotly_chart(drawdown_fig(bt), use_container_width=True)
            st.caption("Nifty 50 is a price index (no dividends), while the stock prices "
                       "are dividend-adjusted, so the benchmark comparison slightly favours the stocks.")