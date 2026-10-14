import numpy as np
import pandas as pd
from src.risk import historical_var_cvar, parametric_var_cvar


def sample_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, 5000))


def test_cvar_is_at_least_var():
    for method in (historical_var_cvar, parametric_var_cvar):
        var, cvar = method(sample_returns(), 0.95)
        assert cvar >= var > 0


def test_higher_confidence_means_higher_var():
    r = sample_returns()
    assert historical_var_cvar(r, 0.99)[0] > historical_var_cvar(r, 0.95)[0]


def test_methods_agree_on_normal_data():
    r = sample_returns()
    hist = historical_var_cvar(r, 0.95)[0]
    param = parametric_var_cvar(r, 0.95)[0]
    assert abs(hist - param) < 0.002