"""tests/test_degeneracy.py — Section 9.1: model-collapse diagnostics."""
import numpy as np
import pytest

from src.eval.degeneracy import compute_degeneracy_for_model


def test_flat_forecast_is_degenerate_by_low_cv():
    n = 200
    sigma2_test = np.full(n, 5.0)  # perfectly flat -> cv = 0
    rng = np.random.default_rng(0)
    eps2_test = rng.exponential(5.0, n)
    res = compute_degeneracy_for_model(
        "FlatModel", "TEST", sigma2_test, eps2_test, mse_constant=1e9,
    )
    assert res["temporal_cv"] == pytest.approx(0.0, abs=1e-9)
    assert res["degenerate"] is True


def test_worse_than_constant_is_degenerate():
    n = 200
    rng = np.random.default_rng(1)
    eps2_test = rng.exponential(5.0, n)
    # A model with plenty of temporal variation but that tracks the target
    # poorly (large noise added) should still be flagged if its MSE is
    # no better than a small mse_constant.
    sigma2_test = eps2_test + rng.normal(0, 10.0, n)
    sigma2_test = np.abs(sigma2_test)
    mse_model = float(np.mean((sigma2_test - eps2_test) ** 2))
    res = compute_degeneracy_for_model(
        "BadModel", "TEST", sigma2_test, eps2_test, mse_constant=mse_model * 0.5,
    )
    assert res["mse_ratio"] >= 1.0
    assert res["degenerate"] is True


def test_good_model_is_not_degenerate():
    n = 500
    rng = np.random.default_rng(2)
    eps2_test = rng.exponential(5.0, n)
    # Track the target closely with real temporal variation and low error.
    sigma2_test = eps2_test * 0.9 + rng.exponential(0.1, n)
    mse_model = float(np.mean((sigma2_test - eps2_test) ** 2))
    res = compute_degeneracy_for_model(
        "GoodModel", "TEST", sigma2_test, eps2_test, mse_constant=mse_model * 5.0,
    )
    assert res["temporal_cv"] > 0.05
    assert res["mse_ratio"] < 1.0
    assert res["degenerate"] is False


def test_cross_seed_relstd_computed_when_per_seed_given():
    n = 100
    rng = np.random.default_rng(3)
    eps2_test = rng.exponential(5.0, n)
    per_seed = np.stack([eps2_test * (0.9 + 0.05 * i) for i in range(5)])
    sigma2_test = per_seed.mean(axis=0)
    mse_model = float(np.mean((sigma2_test - eps2_test) ** 2))
    res = compute_degeneracy_for_model(
        "MultiSeedModel", "TEST", sigma2_test, eps2_test,
        mse_constant=mse_model * 10.0, sigma2_per_seed=per_seed,
    )
    assert res["cross_seed_mse_relstd"] is not None
    assert np.isfinite(res["cross_seed_mse_relstd"])


def test_no_per_seed_data_gives_none_relstd():
    n = 100
    rng = np.random.default_rng(4)
    eps2_test = rng.exponential(5.0, n)
    sigma2_test = eps2_test * 0.9
    res = compute_degeneracy_for_model(
        "SingleEstimateModel", "TEST", sigma2_test, eps2_test, mse_constant=1e9,
    )
    assert res["cross_seed_mse_relstd"] is None
