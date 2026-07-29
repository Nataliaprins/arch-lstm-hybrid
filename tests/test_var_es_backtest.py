"""tests/test_var_es_backtest.py — Section 9.5: VaR/ES backtests."""
import numpy as np
import pytest

from src.eval.var_es_backtest import (
    compute_var_t,
    compute_es_normal,
    compute_es_t,
    hit_sequence,
    kupiec_test,
    christoffersen_test,
    acerbi_szekely_z2,
    run_backtest,
)


def test_es_normal_exceeds_var_normal():
    """ES must always be at least as extreme as VaR at the same level (coherent risk measure)."""
    from src.eval.var_es_backtest import compute_var_normal
    sigma2 = np.full(50, 4.0)
    for alpha in (0.01, 0.025, 0.05):
        var = compute_var_normal(sigma2, alpha)
        es = compute_es_normal(sigma2, alpha)
        assert np.all(es >= var), f"ES < VaR at alpha={alpha} (should never happen)"


def test_es_t_exceeds_var_t():
    sigma2 = np.full(50, 4.0)
    for alpha in (0.01, 0.025, 0.05):
        var = compute_var_t(sigma2, nu=5, alpha=alpha)
        es = compute_es_t(sigma2, nu=5, alpha=alpha)
        assert np.all(es >= var), f"Student-t ES < VaR at alpha={alpha}"


def test_es_normal_scales_with_sigma():
    sigma2_a = np.array([1.0])
    sigma2_b = np.array([4.0])  # 2x the std
    es_a = compute_es_normal(sigma2_a, 0.01)
    es_b = compute_es_normal(sigma2_b, 0.01)
    assert es_b[0] == pytest.approx(2 * es_a[0], rel=1e-6)


def test_christoffersen_recovers_correct_coverage_high_p():
    """Hits drawn at exactly the nominal rate, independently, should not reject."""
    rng = np.random.default_rng(0)
    alpha = 0.01
    T = 5000
    hits = (rng.uniform(size=T) < alpha).astype(float)
    res = christoffersen_test(hits, alpha)
    assert res["p_uc"] > 0.01
    assert res["p_cc"] > 0.01


def test_christoffersen_flags_clustered_exceptions():
    """Strongly clustered hits (violating independence) should give a low p_ind."""
    T = 2000
    hits = np.zeros(T)
    # Place exceptions in tight clusters instead of scattered independently.
    for start in range(0, T, 200):
        hits[start:start + 15] = 1.0
    res = christoffersen_test(hits, alpha=0.01)
    assert res["p_ind"] < 0.05


def test_acerbi_szekely_z2_near_zero_for_correct_model():
    """Returns drawn from the exact model distribution should give Z2 close to 0 and a high p-value."""
    rng = np.random.default_rng(1)
    T = 3000
    nu = 6.0
    sigma2 = np.full(T, 4.0)
    scale = np.sqrt(sigma2 * (nu - 2) / nu)
    returns = rng.standard_t(nu, T) * scale
    alpha = 0.01
    var = compute_var_t(sigma2, nu=nu, alpha=alpha)
    es = compute_es_t(sigma2, nu=nu, alpha=alpha)

    res = acerbi_szekely_z2(returns, var, es, sigma2, nu=nu, alpha=alpha, n_sim=500, seed=1)
    assert abs(res["Z2"]) < 0.5
    assert res["p_value"] > 0.05


def test_acerbi_szekely_z2_flags_underestimated_risk():
    """A model whose ES/VaR are far too small (true vol much higher) should be flagged."""
    rng = np.random.default_rng(2)
    T = 1000
    nu = 6.0
    true_sigma2 = np.full(T, 20.0)   # actual variance much higher than modeled
    model_sigma2 = np.full(T, 4.0)   # underestimated by the model
    scale_true = np.sqrt(true_sigma2 * (nu - 2) / nu)
    returns = rng.standard_t(nu, T) * scale_true
    alpha = 0.01
    var = compute_var_t(model_sigma2, nu=nu, alpha=alpha)
    es = compute_es_t(model_sigma2, nu=nu, alpha=alpha)

    res = acerbi_szekely_z2(returns, var, es, model_sigma2, nu=nu, alpha=alpha, n_sim=500, seed=2)
    assert res["p_value"] < 0.05
    assert res["Z2"] > 0  # realized tail losses worse than the model's ES implies


def test_run_backtest_includes_es_backtest_block():
    rng = np.random.default_rng(3)
    n = 300
    eps = rng.standard_t(5, n) * 2
    sigma2 = np.full(n, 4.0) + rng.exponential(0.5, n)
    res = run_backtest(eps, sigma2, nu=5, levels=[0.99], n_sim_es=200)
    assert "es_backtest" in res["0.99"]
    assert "Z2" in res["0.99"]["es_backtest"]
    assert "p_value" in res["0.99"]["es_backtest"]
    assert res["0.99"]["es_t_mean"] >= res["0.99"]["var_t_mean"]
