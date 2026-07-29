"""tests/test_losses.py — Unit tests for the hybrid Student-t loss."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.losses.hybrid_student_t import (
    hybrid_loss_numpy,
    student_t_nll_per_obs,
    student_t_ll_total,
    make_hybrid_loss,
    make_hybrid_loss_variable_nu,
    make_variance_mse_loss,
    sigma2_from_log_var,
    compute_loss_scales,
    effective_lambda,
    inv_softplus,
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
    """
    Keras hybrid loss ≈ NumPy version (λ=0.5, ν=4). Section 3: the Keras
    loss's y_pred is u_t = log σ̂²_t (raw model output, no activation), so
    we feed log(SIGMA2) and compare against hybrid_loss_numpy(..., SIGMA2).
    """
    import tensorflow as tf
    lam, nu = 0.5, 4
    keras_fn = make_hybrid_loss(nu=nu, lam=lam)
    y_true = tf.constant(EPS2[:20], dtype=tf.float32)
    u_pred = tf.constant(np.log(SIGMA2[:20]), dtype=tf.float32)
    k_loss = float(keras_fn(y_true, u_pred).numpy())
    n_loss = hybrid_loss_numpy(EPS2[:20], SIGMA2[:20], nu=nu, lam=lam)
    assert abs(k_loss - n_loss) < 1e-4, f"Keras={k_loss}  NumPy={n_loss}"


def test_variance_mse_loss_matches_numpy():
    """make_variance_mse_loss(u_t) ≈ MSE(exp(u_t), eps2) in NumPy."""
    import tensorflow as tf
    keras_fn = make_variance_mse_loss()
    u_pred = tf.constant(np.log(SIGMA2), dtype=tf.float32)
    y_true = tf.constant(EPS2, dtype=tf.float32)
    k_loss = float(keras_fn(y_true, u_pred).numpy())
    n_loss = float(np.mean((SIGMA2 - EPS2) ** 2))
    assert abs(k_loss - n_loss) < 1e-3, f"Keras={k_loss}  NumPy={n_loss}"


def test_sigma2_from_log_var_roundtrip():
    """sigma2_from_log_var(log(sigma2)) recovers sigma2 within clip range."""
    u = np.log(SIGMA2)
    recovered = sigma2_from_log_var(u)
    assert np.allclose(recovered, SIGMA2, rtol=1e-5)


def test_sigma2_from_log_var_clips_extremes():
    """Extreme u_t (as could arise early in training) must not overflow/vanish."""
    u = np.array([-1e6, 1e6, 0.0])
    sigma2 = sigma2_from_log_var(u)
    assert np.all(np.isfinite(sigma2))
    assert sigma2[0] > 0.0
    assert sigma2[1] < np.inf


def test_hybrid_loss_finite_for_extreme_log_var():
    """The Keras hybrid loss must stay finite even for wild u_t (pre-clip)."""
    import tensorflow as tf
    keras_fn = make_hybrid_loss(nu=5, lam=0.5)
    y_true = tf.constant(EPS2[:10], dtype=tf.float32)
    u_pred = tf.constant([1e6, -1e6, 30.0, -30.0, 0.0] * 2, dtype=tf.float32)
    loss = float(keras_fn(y_true, u_pred).numpy())
    assert np.isfinite(loss)


def test_positivity_floor():
    """Loss is finite even when sigma2 contains near-zeros."""
    small_sigma2 = np.full(10, 1e-15)
    eps2_test    = np.ones(10)
    loss = hybrid_loss_numpy(eps2_test, small_sigma2, nu=5, lam=0.5)
    assert np.isfinite(loss)


def test_unnormalized_scales_reproduce_old_behavior():
    """s_sse=s_t=1.0 (default) must be exactly the pre-Section-2 loss."""
    old = float(np.mean((SIGMA2 - EPS2) ** 2)) * 0.5 + float(np.mean(
        0.5 * np.log(SIGMA2) + 0.5 * 6.0 * np.log1p(EPS2 / (SIGMA2 * 3.0))
    )) * 0.5
    new = hybrid_loss_numpy(EPS2, SIGMA2, nu=5, lam=0.5)
    assert abs(old - new) < 1e-8


def test_effective_lambda_identity_without_normalization():
    """With s_sse = s_t = 1, lam_effective must equal the nominal lam."""
    for lam in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert abs(effective_lambda(lam, 1.0, 1.0) - lam) < 1e-12


# ── Section 2: loss-normalization test across the four project series ──────
_SERIES = ["BTC-USD", "ETH-USD", "DJIA", "SP500"]
_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_train_eps2(series: str) -> np.ndarray:
    path = _PROCESSED_DIR / series / "train_eps2.csv"
    return pd.read_csv(path, index_col=0).iloc[:, 0].values.astype(float)


@pytest.mark.parametrize("series", _SERIES)
def test_normalized_loss_terms_same_order_of_magnitude(series):
    """
    With loss.normalize=true, (1-lam)-weighted L_SSE and lam-weighted L_t,
    evaluated on the training split with a realistic (non-trivial, non-
    perfect) forecast, must be within one order of magnitude of each other
    in every one of the four project series. This is the regression test
    for the cross-market inconmensurability documented in the brief
    (raw ratio ~99x in BTC, ~290x in ETH before normalization).
    """
    eps2_train = _load_train_eps2(series)
    scales = compute_loss_scales(eps2_train, nu_ref=5.0)

    # Non-trivial in-sample forecast: lag-1 persistence (not derived from
    # the scale computation itself, so this isn't a tautological check).
    sigma2_naive = np.empty_like(eps2_train)
    sigma2_naive[0] = eps2_train.mean()
    sigma2_naive[1:] = eps2_train[:-1]
    sigma2_naive = np.maximum(sigma2_naive, 1e-8)

    nu, nu_m2 = 5.0, 3.0
    L_sse_raw = float(np.mean((sigma2_naive - eps2_train) ** 2))
    L_t_raw = float(np.mean(
        0.5 * np.log(sigma2_naive)
        + 0.5 * (nu + 1) * np.log1p(eps2_train / (sigma2_naive * nu_m2))
    ))

    term_sse = L_sse_raw / scales["s_sse"]
    term_t   = abs(L_t_raw) / scales["s_t"]

    ratio = term_sse / term_t
    assert 0.1 <= ratio <= 10, (
        f"[{series}] normalized terms differ by more than one order of "
        f"magnitude: L_SSE/s_sse={term_sse:.4g}  |L_t|/s_t={term_t:.4g}  "
        f"ratio={ratio:.4g}"
    )


# ── Section 8: learnable nu ─────────────────────────────────────────────────

def test_inv_softplus_roundtrip():
    for y in (0.01, 0.5, 2.0, 6.0, 50.0):
        rho = inv_softplus(y)
        assert abs(np.log1p(np.exp(rho)) - y) < 1e-5


def test_make_hybrid_loss_variable_nu_matches_fixed_nu_at_same_value():
    """A variable-nu loss evaluated at nu=5 must equal the fixed-nu loss at nu=5."""
    import tensorflow as tf

    lam, nu_val = 0.5, 5.0
    fixed_fn = make_hybrid_loss(nu=nu_val, lam=lam)
    variable_fn = make_hybrid_loss_variable_nu(nu_fn=lambda: tf.constant(nu_val), lam=lam)

    y_true = tf.constant(EPS2[:20], dtype=tf.float32)
    u_pred = tf.constant(np.log(SIGMA2[:20]), dtype=tf.float32)

    fixed_loss = float(fixed_fn(y_true, u_pred).numpy())
    variable_loss = float(variable_fn(y_true, u_pred).numpy())
    assert abs(fixed_loss - variable_loss) < 1e-5


def test_make_hybrid_loss_variable_nu_gradient_flows_to_nu():
    """The variable-nu loss must have a non-zero gradient w.r.t. the variable nu is derived from."""
    import tensorflow as tf

    rho = tf.Variable(inv_softplus(3.0), dtype=tf.float32)
    loss_fn = make_hybrid_loss_variable_nu(nu_fn=lambda: 2.0 + tf.nn.softplus(rho), lam=1.0)
    y_true = tf.constant(EPS2, dtype=tf.float32)
    u_pred = tf.constant(np.log(SIGMA2), dtype=tf.float32)

    with tf.GradientTape() as tape:
        loss = loss_fn(y_true, u_pred)
    grad = tape.gradient(loss, rho)
    assert grad is not None
    assert np.isfinite(float(grad.numpy()))
    assert float(grad.numpy()) != 0.0
