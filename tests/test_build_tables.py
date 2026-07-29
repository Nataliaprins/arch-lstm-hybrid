"""tests/test_build_tables.py — spot checks on reporting helpers."""
import numpy as np
import pandas as pd
import pytest

from src.reporting.build_tables import (
    _fisher_combine,
    _save_tex,
    _oos_row,
    build_table8,
)


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


# ── Section 9.6: panel rows rendered as \multicolumn, not repeated text ────

def test_panel_rows_render_as_multicolumn_not_repeated_text(tmp_path):
    df = pd.DataFrame(
        {"MSE": ["—", "1.2345", "—", "2.3456"],
         "QLIKE": ["—", "0.5", "—", "0.6"]},
        index=["Panel A", "ModelX", "Panel B", "ModelY"],
    )
    path = tmp_path / "test.tex"
    _save_tex(df, path, "Test Table", "tab:test", panel_rows={"Panel A", "Panel B"})
    content = path.read_text()

    assert "\\multicolumn{3}{l}{\\textbf{Panel A}}" in content
    assert "\\multicolumn{3}{l}{\\textbf{Panel B}}" in content
    # The old bug: "Panel A & --- Panel A --- & ..." repeated across cells.
    assert "--- Panel A ---" not in content
    assert "--- Panel B ---" not in content
    # Real data rows must still render normally.
    assert "ModelX & 1.2345 & 0.5" in content
    assert "ModelY & 2.3456 & 0.6" in content


# ── Section 9.6: Delta_MSE completeness (only GARCH(1,1) itself is "--") ───

def test_oos_row_delta_mse_filled_for_garch_named_rivals():
    """
    Previously any model whose NAME merely contained the substring
    "GARCH" (EGARCH, GJR-GARCH, FIGARCH, SVR-GARCH, NN-GARCH) had its
    Delta_MSE blanked, even with a real computed value. Only the
    GARCH(1,1) reference row itself should show "--".
    """
    mdata = {"metrics": {"MSE": 1.0, "RMSE": 1.0, "MAE": 1.0, "R2": 0.1,
                          "QLIKE": 1.0, "LL_t_OOS": -1.0, "Delta_MSE": -3.21}}
    for model in ["EGARCH(1,1)", "GJR-GARCH(1,1)", "FIGARCH(1,d,1)", "SVR-GARCH", "NN-GARCH"]:
        row = _oos_row(model, mdata)
        assert row["Delta_MSE"] == "-3.21\\%", f"{model} should show its real Delta_MSE"


def test_oos_row_delta_mse_blanked_only_for_garch_11_itself():
    mdata = {"metrics": {"MSE": 1.0, "RMSE": 1.0, "MAE": 1.0, "R2": 0.1,
                          "QLIKE": 1.0, "LL_t_OOS": -1.0, "Delta_MSE": 0.0}}
    row = _oos_row("GARCH(1,1)", mdata)
    assert row["Delta_MSE"] == "—"


# ── Section 9.6: Table 8 win-direction markers + GARCH(1,1) row ────────────

def test_table8_garch_row_shows_dash_not_zero(tmp_path):
    all_results = {
        "SERIES1": {
            "GARCH(1,1)": {"metrics": {"Delta_MSE": 0.0, "Delta_MAE": 0.0}, "dm_qlike": {}},
        },
    }
    build_table8(all_results, ["SERIES1"], tmp_path)
    content = (tmp_path / "Table8_CrossMarket_Summary.csv").read_text()
    assert "+0.00" not in content
    lines = [l for l in content.splitlines() if "GARCH(1,1)" in l]
    assert lines and "—" in lines[0]


def test_table8_marks_proposed_win_and_rival_win_differently(tmp_path):
    all_results = {
        "SERIES1": {
            "GARCH(1,1)": {"metrics": {"Delta_MSE": 0.0, "Delta_MAE": 0.0}, "dm_qlike": {}},
            "RivalLoses": {
                "metrics": {"Delta_MSE": 10.0, "Delta_MAE": 5.0},
                "dm_qlike": {"reject": True, "DM_stat": 3.0},   # positive -> proposed wins
            },
            "RivalWins": {
                "metrics": {"Delta_MSE": -10.0, "Delta_MAE": -5.0},
                "dm_qlike": {"reject": True, "DM_stat": -3.0},  # negative -> rival wins
            },
            "Inconclusive": {
                "metrics": {"Delta_MSE": 1.0, "Delta_MAE": 1.0},
                "dm_qlike": {"reject": False, "DM_stat": 0.5},
            },
        },
    }
    build_table8(all_results, ["SERIES1"], tmp_path)
    content = (tmp_path / "Table8_CrossMarket_Summary.csv").read_text()

    rival_loses_line = next(l for l in content.splitlines() if l.startswith("RivalLoses"))
    rival_wins_line = next(l for l in content.splitlines() if l.startswith("RivalWins"))
    inconclusive_line = next(l for l in content.splitlines() if l.startswith("Inconclusive"))

    assert "textbf" in rival_loses_line and "underline" not in rival_loses_line
    assert "underline" in rival_wins_line and "textbf" not in rival_wins_line
    assert "textbf" not in inconclusive_line and "underline" not in inconclusive_line
