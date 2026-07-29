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
