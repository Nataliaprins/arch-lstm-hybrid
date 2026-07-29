"""tests/test_build_tables.py — spot checks on reporting helpers."""
import numpy as np
import pytest

from src.reporting.build_tables import _fisher_combine


def test_fisher_combine_all_significant_gives_small_combined_p():
    x2, p = _fisher_combine([0.001, 0.002, 0.0005, 0.001])
    assert p < 0.001


def test_fisher_combine_all_uniform_null_gives_large_combined_p():
    """Four p-values right at 0.5 (perfectly consistent with H0) should combine to a large p."""
    x2, p = _fisher_combine([0.5, 0.5, 0.5, 0.5])
    assert p > 0.3


def test_fisher_combine_is_not_the_mean():
    """
    The whole point of Fisher's method: it must NOT reduce to (or closely
    track) the arithmetic mean -- e.g. three very small p-values and one
    p=1 should still combine to something small (strong joint evidence
    against H0), whereas the mean would be dragged close to 0.25-ish and
    hide the signal.
    """
    p_values = [0.001, 0.001, 0.001, 1.0]
    mean_p = np.mean(p_values)
    _, combined_p = _fisher_combine(p_values)
    assert combined_p < 0.01
    assert combined_p != pytest.approx(mean_p, rel=0.5)


def test_fisher_combine_empty_or_invalid_gives_nan():
    x2, p = _fisher_combine([])
    assert np.isnan(x2) and np.isnan(p)
    x2, p = _fisher_combine([None, float("nan")])
    assert np.isnan(x2) and np.isnan(p)


def test_fisher_combine_ignores_invalid_entries():
    x2_a, p_a = _fisher_combine([0.01, 0.02, 0.03])
    x2_b, p_b = _fisher_combine([0.01, 0.02, 0.03, None, float("nan")])
    assert p_a == pytest.approx(p_b)
