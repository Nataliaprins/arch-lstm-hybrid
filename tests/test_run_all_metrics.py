"""tests/test_run_all_metrics.py — Section 9.7: variance floor / broken-model handling."""
import numpy as np
import pandas as pd
import pytest

from src.eval.run_all_metrics import _load_sigma2, VARIANCE_FLOOR_FACTOR


def test_load_sigma2_prefers_npy(tmp_path):
    series_dir = tmp_path / "SERIES1"
    series_dir.mkdir()
    np.save(series_dir / "sigma2_test.npy", np.array([1.0, 2.0, 3.0]))
    pd.DataFrame({"sigma2_test": [9.0, 9.0, 9.0]}).to_csv(series_dir / "sigma2_test.csv", index=False)

    result = _load_sigma2(tmp_path, "SERIES1")
    assert np.allclose(result, [1.0, 2.0, 3.0])


def test_load_sigma2_falls_back_to_csv(tmp_path):
    """
    Section 9.7: R/msgarch.R saves sigma2_test.csv, not .npy -- MSGARCH
    never appeared in any results table because this loader previously
    only ever checked for the .npy file.
    """
    series_dir = tmp_path / "SERIES1"
    series_dir.mkdir()
    pd.DataFrame({"sigma2_test": [1.5, 2.5, 3.5]}).to_csv(series_dir / "sigma2_test.csv", index=False)

    result = _load_sigma2(tmp_path, "SERIES1")
    assert result is not None
    assert np.allclose(result, [1.5, 2.5, 3.5])


def test_load_sigma2_returns_none_when_neither_exists(tmp_path):
    series_dir = tmp_path / "SERIES1"
    series_dir.mkdir()
    assert _load_sigma2(tmp_path, "SERIES1") is None


def test_variance_floor_factor_is_small_but_positive():
    """Sanity guard: the floor must be a small positive fraction of sigma2_train, not 0 or huge."""
    assert 0 < VARIANCE_FLOOR_FACTOR < 1e-3
