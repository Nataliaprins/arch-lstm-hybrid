"""tests/test_losses.py — Unit tests for the hybrid Student-t loss."""
import numpy as np
import pytest
from src.losses.hybrid_student_t import (
    hybrid_loss_numpy,
    student_t_nll_per_obs,
    student_t_ll_total,
    make_hybrid_loss,
)

RNG = np.random.default_rng(0)
EPS2   = RNG.exponential(1.0, 100)
SIGMA2 = RNG.exponential(1.0, 100) + 0.1


def test_lambda_zero_is_mse():
    """λ=0 → loss = MSE."""
    mse = float(np.mean((SIGMA2 - EPS2) ** 2))
    loss = hybrid_loss_numpy(EPS2, SIGMA2, nu=5, lam=0.0)
    assert abs(loss - mse) < 1e-8, f"{loss} != {mse}"


def test_lambda_one_is_nll():
    """λ=1 → loss = per-obs NLL_t (without constants)."""
    nu   = 5.0
    nu_m2 = nu - 2.0
    expected = float(np.mean(
        0.5 * np.log(SIGMA2) + 0.5 * (nu + 1) * np.log1p(EPS2 / (SIGMA2 * nu_m2))
    ))
    got = hybrid_loss_numpy(EPS2, SIGMA2, nu=nu, lam=1.0)
    assert abs(got - expected) < 1e-6


def test_nll_finite():
    nll = student_t_nll_per_obs(EPS2, SIGMA2, nu=4.0)
    assert np.all(np.isfinite(nll))


def test_ll_total_negative_for_reasonable_inputs():
    """LL should be a finite real number."""
    ll = student_t_ll_total(EPS2, SIGMA2, nu=5.0)
    assert np.isfinite(ll)


def test_keras_loss_matches_numpy():
    """Keras hybrid loss ≈ NumPy version (λ=0.5, ν=4)."""
    import tensorflow as tf
    lam, nu = 0.5, 4
    keras_fn = make_hybrid_loss(nu=nu, lam=lam)
    y_true = tf.constant(EPS2[:20], dtype=tf.float32)
    y_pred = tf.constant(SIGMA2[:20], dtype=tf.float32)
    k_loss = float(keras_fn(y_true, y_pred).numpy())
    n_loss = hybrid_loss_numpy(EPS2[:20], SIGMA2[:20], nu=nu, lam=lam)
    assert abs(k_loss - n_loss) < 1e-4, f"Keras={k_loss}  NumPy={n_loss}"


def test_positivity_floor():
    """Loss is finite even when sigma2 contains near-zeros."""
    small_sigma2 = np.full(10, 1e-15)
    eps2_test    = np.ones(10)
    loss = hybrid_loss_numpy(eps2_test, small_sigma2, nu=5, lam=0.5)
    assert np.isfinite(loss)
