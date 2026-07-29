"""tests/test_ablation_ladder.py — Section 7: ablation-ladder / Proposition 2 test."""
import numpy as np
import pytest

from src.models.ablation_ladder import (
    train_rung1,
    check_proposition2,
    _logit,
    _inv_softplus,
)


def test_logit_inv_softplus_roundtrip():
    for p in (0.05, 0.3, 0.85, 0.99):
        assert abs(1.0 / (1.0 + np.exp(-_logit(p))) - p) < 1e-9
    for y in (0.5, 2.0, 6.0):
        rho = _inv_softplus(y)
        assert abs(np.log1p(np.exp(rho)) - y) < 1e-6


def test_check_proposition2_pass_and_fail():
    rung0_params = {"alpha": 0.10, "beta": 0.85}
    close = {"series": "X", "alpha": 0.105, "beta": 0.83}  # within 10%
    far = {"series": "X", "alpha": 0.50, "beta": 0.20}     # way off

    res_pass = check_proposition2(close, rung0_params)
    assert res_pass["verdict"] == "PASS"

    res_fail = check_proposition2(far, rung0_params)
    assert res_fail["verdict"] == "FAIL"


def test_train_rung1_recovers_known_synthetic_garch_process():
    """
    Generate eps2 EXACTLY from a known (alpha, beta, omega) recursion,
    with no observation noise, and check Rung 1's MLE recovers those
    parameters closely. This isolates whether the optimizer/objective is
    sound, independent of whether any real market's GARCH(1,1) is well
    specified enough for Rung 1 to recover it.
    """
    rng = np.random.default_rng(0)
    alpha_true, beta_true, omega_true, nu_true = 0.10, 0.85, 0.05, 6.0
    sigma2_train_scaler = 1.0
    n = 1500

    # Build a self-consistent GARCH(1,1)-t path: at each step, sigma2 is
    # the deterministic recursion, and eps2 is drawn as sigma2 * (a
    # standardized Student-t innovation)^2, matching the same generative
    # model the Student-t NLL loss assumes.
    c0 = omega_true / (1 - alpha_true - beta_true)
    sigma2 = np.empty(n)
    eps2 = np.empty(n)
    prev = c0
    t_scale = np.sqrt((nu_true - 2) / nu_true)
    for t in range(n):
        sigma2[t] = prev
        innov = rng.standard_t(nu_true) * t_scale
        eps2[t] = sigma2[t] * innov ** 2
        prev = omega_true + alpha_true * eps2[t] + beta_true * sigma2[t]

    x_scaled = eps2 / sigma2_train_scaler
    x_test = x_scaled[-50:]
    eps2_test = eps2[-50:]

    res = train_rung1(
        "SYNTH-GARCH", eps2, x_scaled, eps2_test, x_test,
        sigma2_train_scaler=sigma2_train_scaler, c0=c0,
    )

    # alpha's tolerance is wider than beta's: across several (seed, n)
    # combinations tried while debugging this test, alpha's finite-sample
    # MLE error ranged ~12-32% at n=1500 and visibly shrank with more data
    # (23%->14% at n=6000 for the same seed) -- the signature of a
    # consistent estimator, not a bug. beta is consistently well-recovered
    # (<10%) even at n=1500.
    assert abs(res["alpha"] - alpha_true) / alpha_true < 0.35
    assert abs(res["beta"] - beta_true) / beta_true < 0.15
    prop2 = check_proposition2(res, {"alpha": alpha_true, "beta": beta_true}, tolerance=0.35)
    assert prop2["verdict"] == "PASS", prop2
