import numpy as np
from src.optimiser import min_variance_portfolio, max_sharpe_portfolio
from src.metrics import portfolio_volatility

COV = np.array([[0.04, 0.006, 0.002],
                [0.006, 0.09, 0.004],
                [0.002, 0.004, 0.01]])
EXP = np.array([0.10, 0.15, 0.06])


def test_min_variance_weights_sum_to_one():
    w = min_variance_portfolio(COV)
    assert abs(w.sum() - 1) < 1e-6


def test_no_short_selling():
    for w in (min_variance_portfolio(COV), max_sharpe_portfolio(EXP, COV)):
        assert (w >= -1e-8).all()


def test_min_variance_beats_equal_weight():
    w = min_variance_portfolio(COV)
    equal = np.array([1 / 3] * 3)
    assert portfolio_volatility(w, COV) <= portfolio_volatility(equal, COV) + 1e-9


def test_max_weight_cap_respected():
    w = max_sharpe_portfolio(EXP, COV, max_weight=0.5)
    assert (w <= 0.5 + 1e-6).all()
    assert abs(w.sum() - 1) < 1e-6