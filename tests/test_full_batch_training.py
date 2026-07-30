"""
tests/test_full_batch_training.py — opt-in full-batch (batch_size=n_train)
training mode for the proposed model, requested to more closely match
classical (deterministic) maximum likelihood than mini-batch SGD.
Must not affect the other 7 architectures' shared batch_size search space.
"""
import numpy as np
import pytest

from src.tuning.tune_and_train import _train_one
from src.models.neural import build_lstm_t_student, build_lstm_sse


def _make_data(n=64, W=6):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, W, 1)).astype("float32")
    y = np.abs(rng.normal(size=n)).astype("float32") + 0.5
    return X, y


def test_full_batch_uses_entire_training_set_as_one_batch(monkeypatch):
    """model.fit must be called with batch_size == len(X_train) when full_batch=True."""
    X, y = _make_data(n=50)
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "nu": 5, "lam": 0.5, "batch_size": 8,
    }

    captured = {}
    import tensorflow as tf
    orig_fit = tf.keras.Model.fit

    def spy_fit(self, *args, **kwargs):
        captured["batch_size"] = kwargs.get("batch_size")
        kwargs["epochs"] = 1  # keep the test fast
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(tf.keras.Model, "fit", spy_fit)

    _train_one(build_lstm_t_student, hp, X, y, X[:10], y[:10],
               seed=0, patience=5, max_epochs=1, full_batch=True)
    assert captured["batch_size"] == 50  # == len(X_train), NOT hp["batch_size"]=8


def test_mini_batch_still_uses_hp_batch_size_by_default(monkeypatch):
    """full_batch=False (default) must be unaffected -- existing behavior for all other models."""
    X, y = _make_data(n=50)
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "batch_size": 8,
    }

    captured = {}
    import tensorflow as tf
    orig_fit = tf.keras.Model.fit

    def spy_fit(self, *args, **kwargs):
        captured["batch_size"] = kwargs.get("batch_size")
        kwargs["epochs"] = 1
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(tf.keras.Model, "fit", spy_fit)

    _train_one(build_lstm_sse, hp, X, y, X[:10], y[:10],
               seed=0, patience=5, max_epochs=1)  # full_batch defaults to False
    assert captured["batch_size"] == 8


def test_full_batch_loss_decreases_over_many_epochs():
    """
    Sanity check that full-batch gradient descent can actually learn
    (loss decreases) given enough epochs -- one gradient step per epoch
    needs many more epochs than mini-batch to make comparable progress.
    """
    X, y = _make_data(n=80, W=6)
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-2, "nu": 5, "lam": 0.5, "batch_size": 999,
    }
    _, hist = _train_one(
        build_lstm_t_student, hp, X, y, X[:16], y[:16],
        seed=0, patience=200, max_epochs=200, full_batch=True,
    )
    train_loss = hist["train_loss"]
    assert len(train_loss) > 10
    # Loss at the end should be meaningfully lower than at the start.
    assert train_loss[-1] < train_loss[0]


def test_full_batch_enabled_gate():
    """
    model.full_batch_training must only activate for model_key ==
    "lstm_t_student", leaving every other architecture's batch_size
    search space (and behavior) completely untouched.
    """
    from src.tuning.tune_and_train import _full_batch_enabled

    cfg_on = {"model": {"full_batch_training": True}}
    cfg_off = {"model": {"full_batch_training": False}}
    cfg_missing = {}

    assert _full_batch_enabled(cfg_on, "lstm_t_student") is True
    assert _full_batch_enabled(cfg_on, "lstm_sse") is False
    assert _full_batch_enabled(cfg_on, "cnn_lstm") is False
    assert _full_batch_enabled(cfg_off, "lstm_t_student") is False
    assert _full_batch_enabled(cfg_missing, "lstm_t_student") is False
