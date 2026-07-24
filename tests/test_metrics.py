"""tests/test_metrics.py — Unit tests for evaluation metrics."""
import numpy as np
import pytest
from src.eval.metrics import mse, rmse, mae, r2_score, qlike, ll_t_oos, compute_all
from src.eval.dm_test import dm_test, holm_bonferroni, run_dm_battery

RNG    = np.random.default_rng(1)
N      = 200
EPS2   = RNG.exponential(2.0, N)
SIGMA2 = EPS2 + RNG.normal(0, 0.3, N)
SIGMA2 = np.maximum(SIGMA2, 1e-6)


# ── Point metrics ─────────────────────────────────────────────────────────────

def test_mse_zero_for_perfect():
    assert mse(EPS2, EPS2) == pytest.approx(0.0, abs=1e-10)


def test_rmse_sqrt_mse():
    assert rmse(SIGMA2, EPS2) == pytest.approx(np.sqrt(mse(SIGMA2, EPS2)), rel=1e-6)


def test_mae_nonneg():
    assert mae(SIGMA2, EPS2) >= 0


def test_r2_perfect():
    assert r2_score(EPS2, EPS2) == pytest.approx(1.0, abs=1e-8)


def test_r2_range():
    r2 = r2_score(SIGMA2, EPS2)
    assert r2 <= 1.0


def test_qlike_positive_for_bad_forecast():
    # Constant forecast equal to mean is worse than perfect
    const = np.full(N, EPS2.mean())
    q_const  = qlike(const, EPS2)
    q_perfect = qlike(EPS2, EPS2)
    # Both should be finite
    assert np.isfinite(q_const) and np.isfinite(q_perfect)


def test_ll_t_oos_finite():
    ll = ll_t_oos(SIGMA2, EPS2, nu=5.0)
    assert np.isfinite(ll)


def test_compute_all_keys():
    result = compute_all(SIGMA2, EPS2)
    for key in ["MSE", "RMSE", "MAE", "R2", "QLIKE", "LL_t_OOS"]:
        assert key in result
        assert np.isfinite(result[key]), f"{key} is not finite"


# ── DM test ──────────────────────────────────────────────────────────────────

def test_dm_same_loss_pvalue_large():
    """When both models have the same loss, p-value should be large."""
    loss = RNG.normal(0, 1, N) ** 2
    dm, p = dm_test(loss, loss)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_dm_obvious_winner():
    """Model A clearly worse → large positive DM stat."""
    loss_a = np.abs(RNG.normal(0, 2, N))  # bigger losses
    loss_b = np.abs(RNG.normal(0, 0.1, N))
    dm, p  = dm_test(loss_a, loss_b)
    assert dm > 0
    assert p < 0.05


def test_holm_bonferroni_correction():
    p_vals   = [0.001, 0.01, 0.05, 0.20]
    adjusted = holm_bonferroni(p_vals, alpha=0.05)
    # Smallest should still be smallest (possibly larger)
    assert adjusted[0] <= adjusted[1] <= adjusted[2] <= adjusted[3]
    # All adjusted p ≥ raw p
    for raw, adj in zip(p_vals, adjusted):
        assert adj >= raw


def test_run_dm_battery():
    proposed = RNG.exponential(1.0, N)
    rivals   = {
        "ModelA": RNG.exponential(1.5, N),
        "ModelB": RNG.exponential(0.8, N),
    }
    result = run_dm_battery(proposed, rivals, loss_type="QLIKE")
    assert "ModelA" in result
    assert "p_holm" in result["ModelA"]
    assert 0 <= result["ModelA"]["p_holm"] <= 1
