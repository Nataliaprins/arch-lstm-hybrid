"""
tests/test_qlike_loss.py — feasibility check (2026-08-01): opt-in QLIKE
(Patton 2011) as a replacement for the SSE term in build_lstm_t_student's
hybrid loss (see src.losses.hybrid_student_t.hybrid_loss_components_sigma2_tf
and src.models.neural._make_learned_nu_model). Motivated by an
outlier-dominance check that found 82-99% of L_sse's per-epoch value came
from the top 1% of observations across every series checked -- QLIKE is
the standard robust alternative for volatility loss functions.
"""
import numpy as np
import pytest

from src.models.neural import build_lstm_t_student
from src.losses.hybrid_student_t import hybrid_loss_components_sigma2_tf, compute_loss_scales


def _hp(**overrides):
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "lam": 0.5, "nu_mode": "learned",
        "nu_rho_init": 1.0,
    }
    hp.update(overrides)
    return hp


def test_use_qlike_off_by_default():
    model = build_lstm_t_student(_hp())
    assert model.use_qlike is False


def test_use_qlike_flag_and_scale_propagate():
    model = build_lstm_t_student(_hp(use_qlike=True, s_qlike=2.5))
    assert model.use_qlike is True
    assert model.s_qlike_val == pytest.approx(2.5)


def test_qlike_component_matches_manual_formula():
    """log(sigma2) + eps2/sigma2, mean over the batch -- same terms as
    src.eval.metrics.qlike (up to the -log(eps2)-1 constants, dropped
    here just like every other term in this loss module)."""
    y_true = np.array([2.0, 4.0, 1.0], dtype="float32")
    y_pred = np.array([1.5, 3.0, 2.0], dtype="float32")
    L_first, _ = hybrid_loss_components_sigma2_tf(y_true, y_pred, nu=5.0, use_qlike=True)
    expected = np.mean(np.log(y_pred) + y_true / y_pred)
    assert float(L_first) == pytest.approx(expected, rel=1e-5)


def test_use_qlike_false_still_returns_sse():
    y_true = np.array([2.0, 4.0, 1.0], dtype="float32")
    y_pred = np.array([1.5, 3.0, 2.0], dtype="float32")
    L_first, _ = hybrid_loss_components_sigma2_tf(y_true, y_pred, nu=5.0, use_qlike=False)
    expected = np.mean((y_pred - y_true) ** 2)
    assert float(L_first) == pytest.approx(expected, rel=1e-5)


def test_compute_loss_scales_includes_s_qlike():
    rng = np.random.default_rng(0)
    eps2_train = np.abs(rng.normal(size=200)) + 0.1
    scales = compute_loss_scales(eps2_train)
    assert "s_qlike" in scales
    assert scales["s_qlike"] > 0


def test_model_still_trains_with_qlike():
    """Smoke test: use_qlike=True must not break model.fit() or produce non-finite loss."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 6, 1)).astype("float32")
    y = np.abs(rng.normal(size=20)).astype("float32") + 0.5
    model = build_lstm_t_student(_hp(use_qlike=True, s_qlike=1.0))
    hist = model.fit(X, y, epochs=3, batch_size=20, verbose=0)
    assert all(np.isfinite(v) for v in hist.history["loss"])
