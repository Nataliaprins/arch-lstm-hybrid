"""tests/test_nu_learning.py — Section 8: learnable / likelihood-searched nu."""
import numpy as np
import pytest

from src.models.neural import build_lstm_t_student, set_seeds
from src.losses.hybrid_student_t import inv_softplus
from src.tuning.tune_and_train import _get_nu_mode, _log_nu_kurtosis_spearman


def test_get_nu_mode_default_is_learned():
    assert _get_nu_mode({}) == "learned"


def test_get_nu_mode_reads_config():
    assert _get_nu_mode({"model": {"nu_mode": "likelihood_search"}}) == "likelihood_search"


def test_get_nu_mode_rejects_unknown():
    with pytest.raises(ValueError):
        _get_nu_mode({"model": {"nu_mode": "validation_mse"}})


def test_build_lstm_t_student_fixed_mode_unchanged():
    """Default (no nu_mode key) still returns a plain functional model, not a LearnedNuModel."""
    hp = {"lstm_units": 4, "dropout": 0.0, "window_size": 10, "nu": 5, "lam": 0.5}
    model = build_lstm_t_student(hp)
    assert type(model).__name__ != "_LearnedNuModel"


def test_build_lstm_t_student_learned_mode_output_shape_unchanged():
    """LearnedNuModel's predict() output shape must match the fixed-nu path exactly."""
    set_seeds(0)
    W, n = 10, 40
    X = np.random.exponential(1.0, (n, W, 1)).astype("float32")
    y = (np.random.exponential(1.0, n) * 3 + 1).astype("float32")

    hp_fixed = {"lstm_units": 4, "dropout": 0.0, "window_size": W, "nu": 5, "lam": 0.5}
    hp_learned = {
        "lstm_units": 4, "dropout": 0.0, "window_size": W, "lam": 0.5,
        "nu_mode": "learned", "nu_rho_init": inv_softplus(5.0 - 2.0),
    }

    m_fixed = build_lstm_t_student(hp_fixed)
    m_learned = build_lstm_t_student(hp_learned)

    pred_fixed = m_fixed.predict(X, verbose=0)
    pred_learned = m_learned.predict(X, verbose=0)
    assert pred_fixed.shape == pred_learned.shape == (n, 1)


def test_build_lstm_t_student_learned_mode_nu_moves_during_training():
    set_seeds(0)
    W, n = 10, 60
    X = np.random.exponential(1.0, (n, W, 1)).astype("float32")
    y = (np.random.exponential(1.0, n) * 3 + 1).astype("float32")

    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": W, "lam": 1.0,
        "learning_rate": 0.05,
        "nu_mode": "learned", "nu_rho_init": inv_softplus(20.0 - 2.0),
    }
    model = build_lstm_t_student(hp)
    nu_before = float(model.nu().numpy())
    assert abs(nu_before - 20.0) < 1e-3

    model.fit(X, y, epochs=10, verbose=0, batch_size=16)
    nu_after = float(model.nu().numpy())
    assert nu_after != pytest.approx(nu_before, abs=1e-6)
    assert 2.0 < nu_after < 500.0


def test_learned_mode_history_loss_is_real_not_zero():
    """
    Regression test for a real bug: Keras 3's base Model.compile() creates
    its own internal loss Mean tracker (present in model.metrics) even
    when compile(loss=...) is never called. _LearnedNuModel's custom
    train_step/test_step used to return a raw {"loss": loss} dict without
    ever updating that tracker -- Keras 3's fit()/evaluate() epoch logs
    are derived from model.metrics, not from a custom train_step's
    returned dict, so history.history["loss"]/["val_loss"] silently
    reported a constant 0.0 every single epoch despite real, non-zero
    gradient updates happening (weights DID change). Since
    EarlyStopping(monitor="val_loss") saw val_loss=0.0 at epoch 1 as
    "best" and every later epoch as "no improvement" (0.0 is never
    strictly < 0.0), this caused every affected training run in the
    whole project to stop after exactly patience+1 epochs, regardless of
    real convergence.
    """
    set_seeds(0)
    W, n = 10, 80
    X = np.random.exponential(1.0, (n, W, 1)).astype("float32")
    y = (np.random.exponential(1.0, n) * 3 + 1).astype("float32")

    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": W, "lam": 1.0,
        "learning_rate": 0.05,
        "nu_mode": "learned", "nu_rho_init": inv_softplus(5.0 - 2.0),
    }
    model = build_lstm_t_student(hp)
    hist = model.fit(X, y, validation_data=(X[:20], y[:20]),
                      epochs=5, verbose=0, batch_size=16, shuffle=False)

    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    assert all(v != 0.0 for v in train_loss), f"train_loss stuck at zero: {train_loss}"
    assert all(v != 0.0 for v in val_loss), f"val_loss stuck at zero: {val_loss}"
    assert all(np.isfinite(v) for v in train_loss + val_loss)


def test_learned_mode_early_stopping_sees_real_progress(monkeypatch):
    """
    With the bug, EarlyStopping(patience=P) always stopped at exactly
    P+1 epochs regardless of max_epochs, because every epoch's val_loss
    was the same (0.0). With the fix, a model that keeps improving past
    that point must be allowed to keep training.
    """
    set_seeds(1)
    W, n = 10, 100
    X = np.random.exponential(1.0, (n, W, 1)).astype("float32")
    y = (np.random.exponential(1.0, n) * 3 + 1).astype("float32")

    hp = {
        "lstm_units": 8, "dropout": 0.0, "window_size": W, "lam": 1.0,
        "learning_rate": 0.02,
        "nu_mode": "learned", "nu_rho_init": inv_softplus(5.0 - 2.0),
    }
    model = build_lstm_t_student(hp)
    import tensorflow as tf
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(X, y, validation_data=(X[:20], y[:20]),
                      epochs=200, verbose=0, batch_size=32,
                      callbacks=[es], shuffle=False)
    # The old bug always stopped at exactly patience+1=6 epochs. A real,
    # still-improving loss surface should run well past that.
    assert len(hist.history["loss"]) > 6


def test_log_nu_kurtosis_spearman_writes_expected_line(tmp_path):
    pairs = [("A", 3.0, 10.0), ("B", 5.0, 6.0), ("C", 8.0, 3.0), ("D", 4.0, 8.0)]
    _log_nu_kurtosis_spearman(tmp_path, pairs)
    log_path = tmp_path / "nu_comparison.log"
    assert log_path.exists()
    content = log_path.read_text()
    assert "spearman_rho" in content
    assert "CROSS-SERIES" in content


def test_log_nu_kurtosis_spearman_skips_with_too_few_series(tmp_path):
    _log_nu_kurtosis_spearman(tmp_path, [("A", 3.0, 10.0)])
    log_path = tmp_path / "nu_comparison.log"
    assert not log_path.exists()
