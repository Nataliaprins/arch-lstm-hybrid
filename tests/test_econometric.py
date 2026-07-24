"""tests/test_econometric.py — Smoke-tests for econometric model classes."""
import numpy as np
import pytest
from src.models.econometric import (
    ARCH1Model, GARCH11Model, EGARCH11Model,
    GJRGARCH11Model, HARModel,
)

RNG       = np.random.default_rng(42)
N_TRAIN   = 300
N_TEST    = 50
TRAIN_EPS = RNG.normal(0, 2, N_TRAIN)   # 100×log-ret scale, roughly N(0,4)
TEST_EPS  = RNG.normal(0, 2, N_TEST)


@pytest.mark.parametrize("ModelCls", [
    ARCH1Model, GARCH11Model, EGARCH11Model, GJRGARCH11Model,
])
def test_fit_predict_shape(ModelCls):
    model = ModelCls()
    model.fit(TRAIN_EPS)
    sigma2 = model.predict(TEST_EPS, TRAIN_EPS)
    assert sigma2.shape == (N_TEST,), f"{ModelCls.name}: wrong shape {sigma2.shape}"
    assert np.all(sigma2 > 0), f"{ModelCls.name}: non-positive sigma2"
    assert np.all(np.isfinite(sigma2)), f"{ModelCls.name}: non-finite sigma2"


def test_har_fit_predict():
    model = HARModel()
    model.fit(TRAIN_EPS)
    sigma2 = model.predict(TEST_EPS, TRAIN_EPS)
    assert sigma2.shape == (N_TEST,)
    assert np.all(sigma2 > 0)
    assert np.all(np.isfinite(sigma2))


def test_garch_params_have_expected_keys():
    model = GARCH11Model()
    model.fit(TRAIN_EPS)
    params = model.get_params()
    assert "omega"   in params
    assert "alpha[1]" in params
    assert "beta[1]"  in params
    assert "nu"       in params


def test_garch_fit_info():
    model = GARCH11Model()
    model.fit(TRAIN_EPS)
    info = model.get_fit_info()
    assert "LL_insample"  in info
    assert "convergence"  in info
    assert np.isfinite(info["LL_insample"])


def test_har_params():
    model = HARModel()
    model.fit(TRAIN_EPS)
    params = model.get_params()
    assert "beta_d" in params
    assert "beta_w" in params
    assert "beta_m" in params
