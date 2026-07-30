"""
tests/test_proposed_model_no_garch.py — build_lstm_t_student no longer
uses any GARCH parameter or initialization; output layer guarantees
sigma2_t > 0 via activation="exponential" instead of a downstream
log-variance transform.
"""
import numpy as np
import pytest

from src.models.neural import build_lstm_t_student, set_seeds
from src.losses.hybrid_student_t import inv_softplus


def _hp(**overrides):
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "nu": 5, "lam": 0.5,
    }
    hp.update(overrides)
    return hp


def test_output_layer_is_exponential_activation():
    model = build_lstm_t_student(_hp())
    out_layer = model.layers[-1]
    assert out_layer.activation.__name__ == "exponential"


def test_predict_output_is_always_positive():
    """Raw predict() output IS sigma2_t directly -- must be >0 for arbitrary inputs,
    including large-magnitude ones that would blow up a naive exp() double-application."""
    set_seeds(0)
    model = build_lstm_t_student(_hp())
    X = np.random.default_rng(1).normal(0, 5, size=(20, 6, 1)).astype("float32")
    pred = model.predict(X, verbose=0)
    assert np.all(pred > 0)
    assert np.all(np.isfinite(pred))


def test_stray_garch_keys_are_silently_ignored_not_used():
    """
    Passing old-style GARCH-init keys (as if from a caller that still
    thinks this model supports init="garch") must not change the
    architecture or crash -- the model no longer reads hp["init"] at all.
    """
    hp_plain = _hp()
    hp_with_stray_garch_keys = _hp(
        init="garch", garch_alpha=0.1, garch_beta=0.85,
        garch_omega=0.05, garch_sigma2_train=1.0,
    )
    m1 = build_lstm_t_student(hp_plain)
    m2 = build_lstm_t_student(hp_with_stray_garch_keys)
    # Same architecture (units, layers) regardless of the stray keys.
    assert [type(l).__name__ for l in m1.layers] == [type(l).__name__ for l in m2.layers]
    assert m1.layers[1].units == m2.layers[1].units == 4


def test_lstm_units_not_forced_to_one():
    """Previously the GARCH-init path forced units=1; that constraint is gone."""
    model = build_lstm_t_student(_hp(lstm_units=32))
    lstm_layer = model.layers[1]
    assert lstm_layer.units == 32


def test_lstm_activation_is_default_tanh_not_linear():
    """No more linear-activation GARCH-init path; standard tanh LSTM cell."""
    model = build_lstm_t_student(_hp())
    lstm_layer = model.layers[1]
    assert lstm_layer.get_config()["activation"] == "tanh"


def test_learned_nu_mode_still_works_without_garch_seed():
    """nu_mode="learned" with a fixed, non-GARCH-derived nu_rho_init (Section 8's
    replacement for the old GARCH-nu_hat-derived seed)."""
    hp = _hp(nu_mode="learned", nu_rho_init=inv_softplus(5.0 - 2.0))
    del hp["nu"]
    model = build_lstm_t_student(hp)
    assert type(model).__name__ == "_LearnedNuModel"
    assert float(model.nu().numpy()) == pytest.approx(5.0, abs=1e-3)


def test_two_glorot_inits_give_different_weights():
    """
    Confirms weights really are randomly initialized (glorot), not copied
    from any fixed/external source -- two different seeds must diverge.
    """
    set_seeds(1)
    m1 = build_lstm_t_student(_hp())
    set_seeds(2)
    m2 = build_lstm_t_student(_hp())
    w1 = m1.layers[1].get_weights()[0]
    w2 = m2.layers[1].get_weights()[0]
    assert not np.allclose(w1, w2)
