"""tests/test_tost.py — Section 9.4: TOST equivalence test on QLIKE."""
import numpy as np
import pytest

from src.eval.dm_test import tost_test, run_tost_battery


def test_identical_losses_are_equivalent():
    rng = np.random.default_rng(0)
    loss = rng.exponential(1.0, 500) + 1.0
    res = tost_test(loss, loss, delta=0.02 * float(np.mean(loss)))
    assert res["d_bar"] == pytest.approx(0.0, abs=1e-10)
    assert res["p_tost"] < 0.05
    assert res["equivalent"] is True


def test_wildly_different_losses_are_not_equivalent():
    rng = np.random.default_rng(1)
    loss_model = rng.exponential(1.0, 500) + 10.0   # much worse
    loss_bench = rng.exponential(1.0, 500) + 1.0
    res = tost_test(loss_model, loss_bench, delta=0.02 * float(np.mean(loss_bench)))
    assert res["equivalent"] is False
    assert res["p_tost"] > 0.05


def test_small_difference_within_margin_is_equivalent():
    rng = np.random.default_rng(2)
    loss_bench = rng.exponential(1.0, 2000) + 5.0
    # Model differs by a tiny, consistent amount well within a 5% margin.
    loss_model = loss_bench + 0.01
    delta = 0.05 * float(np.mean(loss_bench))
    res = tost_test(loss_model, loss_bench, delta=delta)
    assert res["equivalent"] is True


def test_tost_is_not_the_mirror_image_of_dm_conclusion():
    """
    Equivalence (small |d|, tight CI around 0) and 'no significant DM
    difference' (large SE, wide CI) are different claims -- TOST should
    reject equivalence for a noisy, inconclusive comparison even though
    a two-sided DM test would fail to reject H0: d=0 there.
    """
    rng = np.random.default_rng(3)
    n = 30  # small n -> high noise -> DM inconclusive
    loss_bench = rng.exponential(1.0, n) + 5.0
    loss_model = loss_bench + rng.normal(0, 5.0, n)  # huge noise, mean ~0
    delta = 0.02 * float(np.mean(loss_bench))
    res = tost_test(loss_model, loss_bench, delta=delta)
    # With so much noise relative to a tight 2% margin, TOST should NOT
    # be able to claim equivalence.
    assert res["equivalent"] is False


def test_run_tost_battery_matches_individual_calls():
    rng = np.random.default_rng(4)
    proposed = rng.exponential(1.0, 300) + 2.0
    rivals = {
        "A": rng.exponential(1.0, 300) + 2.0,
        "B": rng.exponential(1.0, 300) + 5.0,
    }
    delta = 0.02 * float(np.mean(proposed))
    battery = run_tost_battery(proposed, rivals, delta=delta)
    assert set(battery.keys()) == {"A", "B"}
    for name, loss in rivals.items():
        expected = tost_test(proposed, loss, delta=delta)
        assert battery[name]["p_tost"] == pytest.approx(expected["p_tost"])
