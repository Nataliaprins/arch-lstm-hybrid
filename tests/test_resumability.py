"""tests/test_resumability.py — Section 12: resuming an OOM-killed `make all`.

`run()` in tune_and_train.py must be able to restart after the OS kills the
process partway through, without re-running (and without silently reusing)
model-steps -- the frozen run must complete exactly once per model-step.
"""
import json
import time

import numpy as np
import pytest

from src.tuning.tune_and_train import (
    _model_step_complete,
    _lambda_sensitivity_complete,
    _load_nu_kurt_from_log,
)


def _write_complete_step(out_dir, n_seeds=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sigma2_test.npy", np.ones(5))
    np.save(out_dir / "sigma2_per_seed.npy", np.ones((n_seeds, 5)))
    with open(out_dir / "timing.json", "w") as fh:
        json.dump({"tune_seconds": 1.0, "train_seconds": 1.0, "n_seeds": n_seeds}, fh)


def test_missing_artifacts_not_complete(tmp_path):
    out_dir = tmp_path / "LSTM-SSE" / "BTC-USD"
    assert not _model_step_complete(out_dir, "lstm_sse", n_seeds=3, min_mtime=0.0)


def test_fresh_complete_step_detected(tmp_path):
    out_dir = tmp_path / "LSTM-SSE" / "BTC-USD"
    _write_complete_step(out_dir, n_seeds=3)
    assert _model_step_complete(out_dir, "lstm_sse", n_seeds=3, min_mtime=0.0)


def test_svr_garch_complete_without_per_seed_file(tmp_path):
    out_dir = tmp_path / "SVR-GARCH" / "BTC-USD"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "sigma2_test.npy", np.ones(5))
    with open(out_dir / "timing.json", "w") as fh:
        json.dump({"fit_seconds": 1.0, "n_seeds": 1}, fh)
    assert _model_step_complete(out_dir, "svr_garch", n_seeds=10, min_mtime=0.0)


def test_stale_pre_freeze_artifact_is_rejected(tmp_path):
    """
    The core bug this module guards against: outputs/models/ is not wiped
    before a frozen run (only .stamps/ is), so a series/model can carry
    complete-looking artifacts from an earlier development run. Anything
    written before the run's own min_mtime cutoff must NOT count as done.
    """
    out_dir = tmp_path / "TCN" / "SP500"
    _write_complete_step(out_dir, n_seeds=10)
    cutoff_in_the_future = time.time() + 3600
    assert not _model_step_complete(out_dir, "tcn", n_seeds=10, min_mtime=cutoff_in_the_future)


def test_incomplete_seed_count_not_complete(tmp_path):
    out_dir = tmp_path / "TCN" / "SP500"
    _write_complete_step(out_dir, n_seeds=1)  # interrupted after 1 of 10 seeds
    assert not _model_step_complete(out_dir, "tcn", n_seeds=10, min_mtime=0.0)


def test_lambda_sensitivity_complete_requires_all_lambdas(tmp_path):
    proposed_out_dir = tmp_path / "LSTM-SSE-t-Student" / "DJIA"
    lam_dir = proposed_out_dir.parent / "lambda_sensitivity"
    lam_dir.mkdir(parents=True)
    cfg = {"lambda_sensitivity": [0.0, 0.5, 1.0]}
    with open(lam_dir / "DJIA_lambda_sensitivity.json", "w") as fh:
        json.dump([{"lambda": 0.0, "mse": 1.0}, {"lambda": 0.5, "mse": 1.0}], fh)
    assert not _lambda_sensitivity_complete(proposed_out_dir, "DJIA", cfg, min_mtime=0.0)

    with open(lam_dir / "DJIA_lambda_sensitivity.json", "w") as fh:
        json.dump(
            [{"lambda": v, "mse": 1.0} for v in cfg["lambda_sensitivity"]], fh
        )
    assert _lambda_sensitivity_complete(proposed_out_dir, "DJIA", cfg, min_mtime=0.0)


def test_load_nu_kurt_from_log_recovers_correct_series(tmp_path):
    log_path = tmp_path / "nu_comparison.log"
    log_path.write_text(
        "BTC-USD  nu_mode=learned  nu_final=3.1663 (mean_over_10_seeds=3.1663)  "
        "nu_hat_garch_t=3.1688  excess_kurtosis_train_eps=11.4791\n"
        "ETH-USD  nu_mode=learned  nu_final=3.3463 (mean_over_10_seeds=3.3463)  "
        "nu_hat_garch_t=3.3496  excess_kurtosis_train_eps=11.0786\n"
    )
    nu, kurt = _load_nu_kurt_from_log(tmp_path, "BTC-USD")
    assert nu == pytest.approx(3.1663)
    assert kurt == pytest.approx(11.4791)
    assert _load_nu_kurt_from_log(tmp_path, "SP500") is None


def test_load_nu_kurt_from_log_missing_file(tmp_path):
    assert _load_nu_kurt_from_log(tmp_path, "BTC-USD") is None
