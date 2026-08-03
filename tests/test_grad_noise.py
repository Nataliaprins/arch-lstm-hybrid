"""
tests/test_grad_noise.py — feasibility check (2026-08-01): opt-in additive
gradient noise (Neelakantan et al. 2015) for build_lstm_t_student's
nu_mode="learned" path, via its custom train_step (see
src.models.neural._make_learned_nu_model). Distinct from adaptive_lr
(which perturbs step size via an LR schedule, not the gradient itself).
"""
import numpy as np
import pytest

from src.models.neural import build_lstm_t_student, set_seeds


def _hp(**overrides):
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "lam": 0.5, "nu_mode": "learned",
        "nu_rho_init": 1.0,
    }
    hp.update(overrides)
    return hp


def test_grad_noise_off_by_default():
    model = build_lstm_t_student(_hp())
    assert model.grad_noise is False


def test_grad_noise_flags_propagate_when_enabled():
    model = build_lstm_t_student(_hp(grad_noise=True, grad_noise_eta=0.7, grad_noise_gamma=0.4))
    assert model.grad_noise is True
    assert model.grad_noise_eta == pytest.approx(0.7)
    assert model.grad_noise_gamma == pytest.approx(0.4)


def test_grad_noise_actually_perturbs_weights():
    """Same seed, same data, same init -- the only difference is grad_noise
    on/off. If the noise is really wired into train_step, one gradient
    step should land at different weights."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6, 1)).astype("float32")
    y = np.abs(rng.normal(size=20)).astype("float32") + 0.5

    set_seeds(0)
    model_off = build_lstm_t_student(_hp(grad_noise=False))
    model_off.fit(X, y, epochs=1, batch_size=20, verbose=0)
    w_off = model_off.get_weights()

    set_seeds(0)
    model_on = build_lstm_t_student(_hp(grad_noise=True, grad_noise_eta=1.0, grad_noise_gamma=0.0))
    model_on.fit(X, y, epochs=1, batch_size=20, verbose=0)
    w_on = model_on.get_weights()

    assert any(
        not np.allclose(a, b) for a, b in zip(w_off, w_on)
    ), "grad_noise=True produced identical weights to grad_noise=False after one step"


def test_model_still_trains_with_grad_noise():
    """Smoke test: grad_noise=True must not break model.fit() or produce non-finite loss."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6, 1)).astype("float32")
    y = np.abs(rng.normal(size=20)).astype("float32") + 0.5
    model = build_lstm_t_student(_hp(grad_noise=True))
    hist = model.fit(X, y, epochs=3, batch_size=20, verbose=0)
    assert all(np.isfinite(v) for v in hist.history["loss"])
