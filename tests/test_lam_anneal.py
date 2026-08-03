"""
tests/test_lam_anneal.py — feasibility check (2026-08-01): opt-in λ
annealing for build_lstm_t_student's nu_mode="learned" path (see
src.models.neural._make_learned_nu_model). Linearly ramps λ from
lam_start to lam_end over lam_anneal_steps optimizer steps instead of
holding it fixed for the whole run -- analogous to KL-annealing in VAEs
(Bowman et al. 2015), used there to fight "posterior collapse", the same
qualitative failure mode as sigma2_t collapsing to a near-constant value
here.
"""
import numpy as np
import pytest

from src.models.neural import build_lstm_t_student


def _hp(**overrides):
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "lam": 0.9, "nu_mode": "learned",
        "nu_rho_init": 1.0,
    }
    hp.update(overrides)
    return hp


def test_lam_anneal_off_by_default():
    model = build_lstm_t_student(_hp())
    assert model.lam_anneal is False


def test_lam_anneal_flags_propagate_when_enabled():
    model = build_lstm_t_student(_hp(lam_anneal=True, lam_start=0.2, lam_end=0.8, lam_anneal_steps=500))
    assert model.lam_anneal is True
    assert model.lam_start_val == pytest.approx(0.2)
    assert model.lam_end_val == pytest.approx(0.8)
    assert model.lam_anneal_steps_val == pytest.approx(500)


def test_lam_end_defaults_to_lam_when_not_given():
    model = build_lstm_t_student(_hp(lam=0.7, lam_anneal=True, lam_start=0.1))
    assert model.lam_end_val == pytest.approx(0.7)


def test_lam_ramps_linearly_from_start_to_end():
    model = build_lstm_t_student(_hp(lam_anneal=True, lam_start=0.1, lam_end=0.9, lam_anneal_steps=100))
    # optimizer.iterations starts at 0 before any train_step
    model.optimizer.iterations.assign(0)
    lam_0 = float(model._current_lam())
    model.optimizer.iterations.assign(50)
    lam_mid = float(model._current_lam())
    model.optimizer.iterations.assign(100)
    lam_end = float(model._current_lam())
    model.optimizer.iterations.assign(500)  # past anneal_steps -- should hold
    lam_past = float(model._current_lam())

    assert lam_0 == pytest.approx(0.1, abs=1e-6)
    assert lam_mid == pytest.approx(0.5, abs=1e-6)
    assert lam_end == pytest.approx(0.9, abs=1e-6)
    assert lam_past == pytest.approx(0.9, abs=1e-6)


def test_model_still_trains_with_lam_anneal():
    """Smoke test: lam_anneal=True must not break model.fit() or produce non-finite loss."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6, 1)).astype("float32")
    y = np.abs(rng.normal(size=20)).astype("float32") + 0.5
    model = build_lstm_t_student(_hp(lam_anneal=True, lam_start=0.1, lam_end=0.9, lam_anneal_steps=5))
    hist = model.fit(X, y, epochs=8, batch_size=20, verbose=0)
    assert all(np.isfinite(v) for v in hist.history["loss"])
