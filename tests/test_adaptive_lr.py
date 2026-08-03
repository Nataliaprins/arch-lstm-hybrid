"""
tests/test_adaptive_lr.py — collapse-fix (2026-07-31, revised 2026-08-01):
opt-in warm-up + plain cosine-decay learning-rate schedule for
build_lstm_t_student under full-batch training (see
src.models.neural._make_adaptive_lr_schedule). Originally used
cosine-decay-*with-restarts*; switched to a monotonic decay because the
periodic LR restarts kept resetting EarlyStopping's patience counter,
making some series' hyperparameter search take far longer than others'
with no offsetting benefit against collapse.
"""
import numpy as np
import pytest

from src.models.neural import build_lstm_t_student, _make_adaptive_lr_schedule


def _hp(**overrides):
    hp = {
        "lstm_units": 4, "dropout": 0.0, "window_size": 6,
        "learning_rate": 1e-3, "nu": 5, "lam": 0.5,
    }
    hp.update(overrides)
    return hp


def test_adaptive_lr_off_by_default_is_a_constant():
    model = build_lstm_t_student(_hp())
    import tensorflow as tf
    # Keras stores the configured LR (schedule or constant) on the
    # optimizer's private _learning_rate; .learning_rate is a live
    # tensor/variable snapshot of its current value either way, so it
    # can't be used to tell the two apart.
    lr = model.optimizer._learning_rate
    assert not isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule)


def test_adaptive_lr_true_uses_a_schedule():
    model = build_lstm_t_student(_hp(adaptive_lr=True))
    import tensorflow as tf
    lr = model.optimizer._learning_rate
    assert isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule)


def test_warmup_ramps_linearly_to_peak():
    schedule = _make_adaptive_lr_schedule({"learning_rate": 1e-3, "lr_warmup_steps": 10})
    lr_start = float(schedule(0))
    lr_mid   = float(schedule(5))
    lr_end   = float(schedule(10))
    assert lr_start == pytest.approx(0.0, abs=1e-9)
    assert lr_mid   == pytest.approx(5e-4, rel=1e-3)
    assert lr_end   == pytest.approx(1e-3, rel=1e-3)


def test_decay_is_monotonic_non_increasing_after_warmup():
    """No restarts: once past warm-up, LR should only decrease (or hold at
    the floor), never jump back up -- the whole point of dropping SGDR was
    to stop val_loss from being periodically kicked back up, which had
    been resetting EarlyStopping's patience counter."""
    schedule = _make_adaptive_lr_schedule({
        "learning_rate": 1e-3, "lr_warmup_steps": 1, "lr_decay_steps": 100, "lr_alpha": 0.0,
    })
    lrs = [float(schedule(s)) for s in range(1, 140)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_decay_reaches_alpha_floor_and_holds():
    schedule = _make_adaptive_lr_schedule({
        "learning_rate": 1e-3, "lr_warmup_steps": 1, "lr_decay_steps": 100, "lr_alpha": 0.05,
    })
    lr_at_floor  = float(schedule(101))
    lr_well_past = float(schedule(500))
    assert lr_at_floor == pytest.approx(0.05e-3, rel=1e-3)
    assert lr_well_past == pytest.approx(0.05e-3, rel=1e-3)


def test_model_still_trains_one_step_with_adaptive_lr():
    """Smoke test: adaptive_lr=True must not break model.fit()."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6, 1)).astype("float32")
    y = np.abs(rng.normal(size=20)).astype("float32") + 0.5
    model = build_lstm_t_student(_hp(adaptive_lr=True))
    hist = model.fit(X, y, epochs=2, batch_size=20, verbose=0)
    assert all(np.isfinite(v) for v in hist.history["loss"])
