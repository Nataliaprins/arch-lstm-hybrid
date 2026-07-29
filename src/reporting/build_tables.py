"""
build_tables.py — Emit Tables 3–11 and 4+A1–A4 in .csv, .tex (booktabs), .docx.

Usage:
    python -m src.reporting.build_tables --config config/config.yaml

Input:  outputs/tables/raw_results.json
        outputs/tables/encompassing_results.json
        outputs/models/<model>/<series>/{params.json, fit_info.json}
Output: outputs/tables/  *.csv  *.tex  *.docx

Table numbering (paper convention)
-----------------------------------
Table 3   — Model roster (static)
Tables 4–7 — OOS performance, one per series
Table 8   — Cross-market Δ% summary
Table 9   — Diebold–Mariano
Table 10  — Forecast-Encompassing
Table 11a–11d — Risk backtests (Kupiec + Christoffersen + VaR + ES), one per series
Table 12  — Cross-market risk summary (Kupiec + Christoffersen + VaR + ES)
Table 4e  — GARCH(1,1) estimation (all four series)
Tables A1–A4 — Full estimation per series (all econometric models)
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _fmt(v, decimals=4, pct=False, bold=False) -> str:
    """Format a scalar for display."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if pct:
        s = f"{v:+.2f}\\%"
    else:
        s = f"{v:.{decimals}f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def _sig_stars(p) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)
    log.info("Saved CSV: %s", path)


def _save_tex(df: pd.DataFrame, path: Path, caption: str, label: str, note: str = "") -> None:
    """Emit a booktabs LaTeX table."""
    n_cols  = len(df.columns)
    col_fmt = "l" + "r" * n_cols
    lines   = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
    ]
    # Header
    header  = " & " + " & ".join(str(c) for c in df.columns) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    # Rows
    for idx, row in df.iterrows():
        cells = str(idx) + " & " + " & ".join(
            str(v) if v is not None else "—" for v in row
        ) + " \\\\"
        lines.append(cells)
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if note:
        lines.append(f"\\\\[3pt]\\footnotesize\\textit{{Note:}} {note}")
    lines.append("\\end{table}")

    path.write_text("\n".join(lines))
    log.info("Saved TEX: %s", path)


def _save_docx(df: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    """Emit a three-line MDPI-style .docx table."""
    try:
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        log.warning("python-docx not available; skipping .docx for %s", path)
        return

    doc   = Document()
    doc.add_heading(title, level=2)

    n_cols = len(df.columns) + 1  # +1 for index
    table  = doc.add_table(rows=1 + len(df), cols=n_cols)
    table.style = "Table Grid"

    # Header row
    hdr = table.rows[0].cells
    hdr[0].text = df.index.name or "Model"
    for j, col in enumerate(df.columns):
        hdr[j + 1].text = str(col)

    # Data rows
    for i, (idx, row) in enumerate(df.iterrows()):
        cells = table.rows[i + 1].cells
        cells[0].text = str(idx)
        for j, v in enumerate(row):
            cells[j + 1].text = str(v) if v is not None else "—"

    # Style: Times New Roman 10pt
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)

    if note:
        p = doc.add_paragraph()
        run = p.add_run(f"Note: {note}")
        run.font.size = Pt(9)
        run.font.italic = True

    doc.save(str(path))
    log.info("Saved DOCX: %s", path)


def _save_all(df: pd.DataFrame, stem: Path, caption: str, label: str, note: str) -> None:
    """Save CSV + TEX + DOCX."""
    _save_csv(df, stem.with_suffix(".csv"))
    _save_tex(df, stem.with_suffix(".tex"), caption, label, note)
    _save_docx(df, stem.with_suffix(".docx"), caption, note)


# ──────────────────────────────────────────────────────────────────────────────
# Table 3 — Model Roster (static)
# ──────────────────────────────────────────────────────────────────────────────

ROSTER_DATA = [
    # Panel A — Econometric
    ("Panel A", "ARCH(1)",         "Econometric", "ARCH(1)-t",            "arch (Python)",    "Benchmark"),
    ("Panel A", "GARCH(1,1)",      "Econometric", "GARCH(1,1)-t",         "arch (Python)",    "Reference benchmark"),
    ("Panel A", "EGARCH(1,1)",     "Econometric", "EGARCH(1,1)-t",        "arch (Python)",    "Leverage effect"),
    ("Panel A", "GJR-GARCH(1,1)",  "Econometric", "GJR-GARCH(1,1)-t",     "arch (Python)",    "Threshold asymmetry"),
    ("Panel A", "FIGARCH(1,d,1)",  "Econometric", "FIGARCH(1,d,1)-t",     "arch (Python)",    "Long memory"),
    ("Panel A", "MSGARCH(1,1)",    "Econometric", "Markov-switching (2 states)", "MSGARCH (R)",  "Markov-switching (rev. req.)"),
    ("Panel A", "HAR",             "Econometric", "HAR-RV (OLS+HAC)",     "statsmodels",      "Heterogeneous volatility"),
    # Panel B — ML / DL
    ("Panel B", "SVR-GARCH",       "ML",          "SVR-RBF on ε² window", "scikit-learn",     "ML baseline"),
    ("Panel B", "NN-GARCH",        "DL",          "Dense net + GARCH prior", "Keras/TF",      "Neural GARCH augmentation (rev. req.)"),
    ("Panel B", "LSTM-SSE",        "DL",          "LSTM + MSE loss",      "Keras/TF",         "Architecture ablation"),
    ("Panel B", "CNN-LSTM",        "DL",          "Conv1D + LSTM",        "Keras/TF",         "DL baseline"),
    ("Panel B", "LSTM-Attention",  "DL",          "LSTM + Bahdanau attn", "Keras/TF",         "DL baseline"),
    ("Panel B", "TCN",             "DL",          "Dilated causal Conv",  "Keras/TF",         "DL baseline"),
    ("Panel B", "Transformer",     "DL",          "Encoder-only transformer", "Keras/TF",     "DL baseline"),
    # Panel C — Proposed
    ("Panel C", "LSTM-SSE-t-Student", "DL (proposed)",
     "LSTM + (1−λ)·MSE + λ·NLL_t", "Keras/TF", "Proposed model"),
]


def build_table3(out_dir: Path) -> None:
    cols    = ["Panel", "Model", "Family", "Specification", "Package", "Role"]
    df      = pd.DataFrame(ROSTER_DATA, columns=cols).set_index("Model")
    note    = ("Returns scaled as 100×log-retornos (pp). All parametric models use "
               "Student-t innovations. Neural models: S=10 seeds, mean ± s.d. reported. "
               "ε²_t as volatility proxy for all evaluations.")
    stem    = out_dir / "Table3_model_roster"
    _save_all(df, stem, "Table 3. Model Roster", "tab:roster", note)


# ──────────────────────────────────────────────────────────────────────────────
# Tables 4–7 — OOS Performance (one per series)
# ──────────────────────────────────────────────────────────────────────────────

PANEL_ORDER = {
    "Panel A": ["ARCH(1)", "GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH(1,1)",
                "FIGARCH(1,d,1)", "MSGARCH(1,1)", "HAR"],
    "Panel B": ["SVR-GARCH", "NN-GARCH", "LSTM-SSE", "CNN-LSTM",
                "LSTM-Attention", "TCN", "Transformer"],
    "Panel C": ["LSTM-SSE-t-Student"],
}

PANEL_MAP = {m: p for p, ms in PANEL_ORDER.items() for m in ms}

OOS_COLS = ["MSE", "RMSE", "MAE", "R2", "QLIKE", "LL_t_OOS", "Delta_MSE", "MCS_90"]


def _oos_row(model: str, mdata: dict) -> dict:
    m = mdata.get("metrics", {})
    s = mdata.get("std", {})

    def _v(key):
        v = m.get(key)
        return v if v is not None else float("nan")

    mse_v  = _v("MSE")
    rmse_v = _v("RMSE")
    mae_v  = _v("MAE")
    r2_v   = _v("R2")
    ql_v   = _v("QLIKE")
    ll_v   = _v("LL_t_OOS")
    dm_v   = _v("Delta_MSE")

    mcs = mdata.get("mcs_90")
    mcs_str = "Yes" if mcs is True else ("No" if mcs is False else "—")

    # Panel B: mean ± std format
    std_mse = s.get("MSE_std")
    std_mae = s.get("MAE_std")
    mse_str = (f"{mse_v:.5f} ± {std_mse:.5f}" if std_mse else _fmt(mse_v, 5))
    mae_str = (f"{mae_v:.5f} ± {std_mae:.5f}" if std_mae else _fmt(mae_v, 5))

    return {
        "MSE":        mse_str,
        "RMSE":       _fmt(rmse_v, 5),
        "MAE":        mae_str,
        "R2":         _fmt(r2_v, 4),
        "QLIKE":      _fmt(ql_v, 5),
        "LL_t_OOS":   _fmt(ll_v, 4),
        "Delta_MSE":  _fmt(dm_v, 2, pct=True) if "GARCH" not in model else "—",
        "MCS_90":     mcs_str,
    }


def _best_col_mask(df_raw: pd.DataFrame, col: str) -> pd.Index:
    """Return index of best (min for most, max for R2/LL) value in a column."""
    try:
        vals = pd.to_numeric(df_raw[col].apply(
            lambda x: x.split("±")[0].strip().replace("%", "").replace("\\%", "").replace("+", "")
            if isinstance(x, str) else x
        ), errors="coerce")
        if col in ("R2", "LL_t_OOS"):
            return df_raw.index[vals == vals.max()]
        else:
            return df_raw.index[vals == vals.min()]
    except Exception:
        return pd.Index([])


def build_table_oos(series: str, results: dict, tnum: int, out_dir: Path) -> None:
    rows = []
    for panel, models in PANEL_ORDER.items():
        panel_header = {"MSE": f"--- {panel} ---"}
        rows.append((panel, panel_header))
        for m in models:
            if m in results:
                rows.append((m, _oos_row(m, results[m])))

    df_raw = pd.DataFrame(
        [r for _, r in rows],
        index=[n for n, _ in rows],
    ).fillna("—")

    # Bold best values per column
    for col in ["MSE", "RMSE", "MAE", "R2", "QLIKE", "LL_t_OOS"]:
        if col not in df_raw.columns:
            continue
        best_idx = _best_col_mask(df_raw, col)
        for idx in best_idx:
            v = df_raw.at[idx, col]
            if isinstance(v, str) and "---" not in v:
                df_raw.at[idx, col] = f"\\textbf{{{v}}}"

    note = (
        f"OOS window: {results.get(list(results)[0], {}).get('n_test', '?')} observations. "
        "Returns in 100×log-ret (pp); proxy ε²_t = centred squared return. "
        "QLIKE = T⁻¹Σ[lnσ̂²_t + ε²_t/σ̂²_t]. "
        "LL_t(OOS): per-obs avg Student-t log-likelihood evaluated OOS (all models). "
        "Δ%%MSE relative to GARCH(1,1); negative = improvement. "
        "MCS 90%% based on block-bootstrap QLIKE (B=10 000, block=20). "
        "Panel B: mean ± s.d. over S=10 seeds. "
        "Standard errors Bollerslev–Wooldridge; *** p<0.01."
    )

    stem = out_dir / f"Table{tnum}_OOS_{series}"
    _save_all(
        df_raw, stem,
        f"Table {tnum}. OOS Forecasting Performance — {series}",
        f"tab:oos_{series.lower()}",
        note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Table 8 — Cross-market Δ% summary
# ──────────────────────────────────────────────────────────────────────────────

def build_table8(all_results: dict, series_list: list[str], out_dir: Path) -> None:
    all_models = sorted({m for s in all_results.values() for m in s})
    # Multi-level columns: series × {Delta_MSE, Delta_MAE}
    col_tuples = [(s, metric) for s in series_list for metric in ["Δ%MSE", "Δ%MAE"]]
    cols       = pd.MultiIndex.from_tuples(col_tuples)

    rows = []
    for model in all_models:
        row = []
        for series in series_list:
            m_res = all_results.get(series, {}).get(model, {})
            metrics = m_res.get("metrics", {})
            dm_holm  = m_res.get("dm_qlike", {})
            bold = dm_holm.get("reject", False)

            dm_pct  = metrics.get("Delta_MSE")
            mae_pct = metrics.get("Delta_MAE")

            def _cell(v):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return "—"
                s = f"{v:+.2f}"
                return f"\\textbf{{{s}}}" if bold else s

            row.extend([_cell(dm_pct), _cell(mae_pct)])
        rows.append(row)

    df = pd.DataFrame(rows, index=all_models, columns=cols)
    note = (
        "Percentage change relative to GARCH(1,1) (reference row marked —). "
        "Bold = statistically significant at 5%% after Holm–Bonferroni correction "
        "on QLIKE-based DM test. Negative = improvement over GARCH(1,1)."
    )
    stem = out_dir / "Table8_CrossMarket_Summary"
    _save_all(df, stem, "Table 8. Cross-Market OOS Performance Summary",
              "tab:cross_market", note)


# ──────────────────────────────────────────────────────────────────────────────
# Table 9 — Diebold–Mariano
# ──────────────────────────────────────────────────────────────────────────────

def build_table9(all_results: dict, series_list: list[str], out_dir: Path) -> None:
    all_models = sorted({
        m for s in all_results.values()
        for m in s if m != "LSTM-SSE-t-Student"
    })

    col_tuples = [(s, stat) for s in series_list for stat in ["DM", "p(Holm)"]]
    cols = pd.MultiIndex.from_tuples(col_tuples)

    rows = []
    for rival in all_models:
        row = []
        for series in series_list:
            dm_res = all_results.get(series, {}).get(rival, {}).get("dm_qlike", {})
            dm_v   = dm_res.get("DM_stat")
            p_holm = dm_res.get("p_holm")
            stars  = _sig_stars(p_holm)
            dm_str = f"{dm_v:.3f}{stars}" if dm_v is not None else "—"
            ph_str = f"{p_holm:.3f}" if p_holm is not None else "—"
            row.extend([dm_str, ph_str])
        rows.append(row)

    df = pd.DataFrame(rows, index=all_models, columns=cols)
    note = (
        "Diebold–Mariano test: LSTM-SSE-t-Student vs. each rival model. "
        "Loss function: QLIKE. HAC SE (Newey–West, bandwidth ⌊4(T/100)^{2/9}⌋). "
        "p(Holm): p-value after Holm–Bonferroni FWER correction per market. "
        "Positive DM stat: rival has higher loss → proposed wins. "
        "Significance: *** p<0.01, ** p<0.05, * p<0.10 (after Holm)."
    )
    stem = out_dir / "Table9_DieboldMariano"
    _save_all(df, stem, "Table 9. Diebold–Mariano Tests", "tab:dm", note)


# ──────────────────────────────────────────────────────────────────────────────
# Table 10 — Forecast-Encompassing Tests
# ──────────────────────────────────────────────────────────────────────────────

def build_table10(encompassing_all: dict, series_list: list[str], out_dir: Path) -> None:
    """
    ε²_t = b0 + b1·σ²_benchmark + b2·σ²_candidate + u_t   (HAC SE).

    One row per (candidate, benchmark) pair; columns show β_candidate
    (with significance stars on p_candidate) and the verdict per series.
    """
    all_pairs = sorted({
        pair for s in encompassing_all.values() for pair in s
    })

    col_tuples = [(s, stat) for s in series_list for stat in ["β_cand", "verdict"]]
    cols = pd.MultiIndex.from_tuples(col_tuples)

    verdict_labels = {
        "candidate_adds_info":  "Adds info",
        "benchmark_sufficient": "Benchmark suff.",
        "both_contribute":      "Both contribute",
        "neither_significant":  "Neither sig.",
    }

    rows = []
    for pair in all_pairs:
        row = []
        for series in series_list:
            res = encompassing_all.get(series, {}).get(pair, {})
            if "error" in res:
                row.extend(["—", "n/a"])
                continue
            b_cand  = res.get("beta_candidate")
            p_cand  = res.get("p_candidate")
            stars   = _sig_stars(p_cand)
            b_str   = f"{b_cand:.3f}{stars}" if b_cand is not None else "—"
            verdict = verdict_labels.get(res.get("verdict"), "—")
            row.extend([b_str, verdict])
        rows.append(row)

    pair_labels = [p.replace("_vs_", " vs. ") for p in all_pairs]
    df = pd.DataFrame(rows, index=pair_labels, columns=cols)

    note = (
        "Forecast-encompassing regression: \\varepsilon^2_t = b_0 + b_1 "
        "\\sigma^2_{benchmark,t} + b_2 \\sigma^2_{candidate,t} + u_t, HAC SE "
        "(Newey--West, bandwidth \\lfloor 4(T/100)^{2/9} \\rfloor). "
        "\\beta_{cand}: coefficient on the candidate forecast (significance "
        "from its own HAC p-value). 'Adds info': candidate significant, "
        "benchmark not — benchmark is redundant given the candidate. "
        "'Both contribute': both significant — complementary information. "
        "'Benchmark suff.': candidate carries no information beyond the "
        "benchmark. 'Neither sig.': inconclusive (often collinearity). "
        "Significance: *** p<0.01, ** p<0.05, * p<0.10."
    )
    stem = out_dir / "Table10_Encompassing"
    _save_all(df, stem, "Table 10. Forecast-Encompassing Tests", "tab:encompassing", note)


# ──────────────────────────────────────────────────────────────────────────────
# Table 11a–11d — Risk backtests (Kupiec + Christoffersen + VaR + ES)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_bool_pass(p_uc: Any) -> str:
    if p_uc is None or (isinstance(p_uc, float) and not np.isfinite(p_uc)):
        return "—"
    return "Yes" if p_uc >= 0.05 else "No"


def _fmt_bool_pass_cc(p_cc: Any) -> str:
    if p_cc is None or (isinstance(p_cc, float) and not np.isfinite(p_cc)):
        return "—"
    return "Yes" if p_cc >= 0.05 else "No"


def _risk_row(m_res: dict, level: str) -> dict:
    ves = m_res.get("var_es", {}).get(level, {})
    st  = ves.get("student_t", {})
    return {
        f"LR_uc@{level}": _fmt(st.get("LR_uc"), decimals=4),
        f"p_uc@{level}": _fmt(st.get("p_uc"), decimals=4),
        f"ExcRate@{level}": _fmt(st.get("exc_rate"), decimals=4),
        f"n_exc@{level}": str(st.get("n_exc", "—")),
        f"KupiecPass@{level}": _fmt_bool_pass(st.get("p_uc")),
        f"LR_ind@{level}": _fmt(st.get("LR_ind"), decimals=4),
        f"p_ind@{level}": _fmt(st.get("p_ind"), decimals=4),
        f"LR_cc@{level}": _fmt(st.get("LR_cc"), decimals=4),
        f"p_cc@{level}": _fmt(st.get("p_cc"), decimals=4),
        f"ChristoffersenPass@{level}": _fmt_bool_pass_cc(st.get("p_cc")),
        f"VaR_t_mean@{level}": _fmt(ves.get("var_t_mean"), decimals=6),
        f"ES_mean@{level}": _fmt(ves.get("es_mean"), decimals=6),
    }


def build_table11_risk(series: str, results: dict, tlabel: str, out_dir: Path) -> None:
    if not results:
        return

    all_models = sorted(results.keys())
    levels = sorted({
        lv
        for m in results.values()
        for lv in m.get("var_es", {}).keys()
    }, key=float, reverse=True)

    rows = []
    for model in all_models:
        m_res = results.get(model, {})
        row = {}
        for lv in levels:
            row.update(_risk_row(m_res, lv))
        rows.append((model, row))

    if not rows:
        return

    df = pd.DataFrame([r for _, r in rows], index=[m for m, _ in rows])
    note = (
        "Student-t VaR backtest over OOS residuals. Kupiec uses the UC LR test "
        "for correct unconditional exceedance rate (H0: hit rate = tail probability). "
        "Christoffersen uses the CC LR test (UC + independence). "
        "KupiecPass = Yes when p_uc >= 0.05; ChristoffersenPass = Yes when p_cc >= 0.05. "
        "VaR_t_mean and ES_mean are averages of model-implied risk forecasts over the OOS window."
    )
    stem = out_dir / f"Table11_{tlabel}_Risk_{series}"
    _save_all(
        df,
        stem,
        f"Table 11{tlabel}. Risk Backtest Summary — {series}",
        f"tab:risk_{series.lower()}",
        note,
    )


def build_table12_risk_summary(all_results: dict, series_list: list[str], out_dir: Path) -> None:
    """Cross-market summary of Kupiec, Christoffersen, VaR and ES by model."""
    all_models = sorted({m for s in all_results.values() for m in s})
    levels = ["0.99", "0.975"]

    rows = []
    for model in all_models:
        rec = {"Model": model}
        pass_count_uc_total = 0
        obs_uc_total = 0
        pass_count_cc_total = 0
        obs_cc_total = 0

        for lv in levels:
            lr_vals = []
            p_vals = []
            lr_ind_vals = []
            p_ind_vals = []
            lr_cc_vals = []
            p_cc_vals = []
            var_vals = []
            es_vals = []
            pass_count_uc_lv = 0
            obs_uc_lv = 0
            pass_count_cc_lv = 0
            obs_cc_lv = 0

            for series in series_list:
                m_res = all_results.get(series, {}).get(model, {})
                ves = m_res.get("var_es", {}).get(lv, {})
                st = ves.get("student_t", {})

                lr_uc = st.get("LR_uc")
                p_uc = st.get("p_uc")
                lr_ind = st.get("LR_ind")
                p_ind = st.get("p_ind")
                lr_cc = st.get("LR_cc")
                p_cc = st.get("p_cc")
                var_m = ves.get("var_t_mean")
                es_m = ves.get("es_mean")

                if isinstance(lr_uc, (int, float)) and np.isfinite(lr_uc):
                    lr_vals.append(float(lr_uc))
                if isinstance(p_uc, (int, float)) and np.isfinite(p_uc):
                    p_vals.append(float(p_uc))
                    obs_uc_lv += 1
                    if p_uc >= 0.05:
                        pass_count_uc_lv += 1
                if isinstance(lr_ind, (int, float)) and np.isfinite(lr_ind):
                    lr_ind_vals.append(float(lr_ind))
                if isinstance(p_ind, (int, float)) and np.isfinite(p_ind):
                    p_ind_vals.append(float(p_ind))
                if isinstance(lr_cc, (int, float)) and np.isfinite(lr_cc):
                    lr_cc_vals.append(float(lr_cc))
                if isinstance(p_cc, (int, float)) and np.isfinite(p_cc):
                    p_cc_vals.append(float(p_cc))
                    obs_cc_lv += 1
                    if p_cc >= 0.05:
                        pass_count_cc_lv += 1
                if isinstance(var_m, (int, float)) and np.isfinite(var_m):
                    var_vals.append(float(var_m))
                if isinstance(es_m, (int, float)) and np.isfinite(es_m):
                    es_vals.append(float(es_m))

            pass_count_uc_total += pass_count_uc_lv
            obs_uc_total += obs_uc_lv
            pass_count_cc_total += pass_count_cc_lv
            obs_cc_total += obs_cc_lv

            rec[f"PassRate_UC@{lv}(%)"] = 100.0 * pass_count_uc_lv / obs_uc_lv if obs_uc_lv > 0 else np.nan
            rec[f"Avg_p_uc@{lv}"] = float(np.mean(p_vals)) if p_vals else np.nan
            rec[f"Avg_LR_uc@{lv}"] = float(np.mean(lr_vals)) if lr_vals else np.nan
            rec[f"PassRate_CC@{lv}(%)"] = 100.0 * pass_count_cc_lv / obs_cc_lv if obs_cc_lv > 0 else np.nan
            rec[f"Avg_p_cc@{lv}"] = float(np.mean(p_cc_vals)) if p_cc_vals else np.nan
            rec[f"Avg_LR_cc@{lv}"] = float(np.mean(lr_cc_vals)) if lr_cc_vals else np.nan
            rec[f"Avg_p_ind@{lv}"] = float(np.mean(p_ind_vals)) if p_ind_vals else np.nan
            rec[f"Avg_LR_ind@{lv}"] = float(np.mean(lr_ind_vals)) if lr_ind_vals else np.nan
            rec[f"Avg_VaR_t@{lv}"] = float(np.mean(var_vals)) if var_vals else np.nan
            rec[f"Avg_ES@{lv}"] = float(np.mean(es_vals)) if es_vals else np.nan

        rec["PassRate_UC_Global(%)"] = 100.0 * pass_count_uc_total / obs_uc_total if obs_uc_total > 0 else np.nan
        rec["PassRate_CC_Global(%)"] = 100.0 * pass_count_cc_total / obs_cc_total if obs_cc_total > 0 else np.nan
        rows.append(rec)

    if not rows:
        return

    df_raw = pd.DataFrame(rows).set_index("Model")
    # Higher pass rates and p-values are better; lower LR stats / VaR / ES are better.
    sort_cols = [
        "PassRate_CC_Global(%)",
        "PassRate_UC_Global(%)",
        "Avg_p_cc@0.99",
        "Avg_p_cc@0.975",
        "Avg_ES@0.99",
        "Avg_ES@0.975",
    ]
    asc = [False, False, False, False, True, True]
    df_raw = df_raw.sort_values(by=sort_cols, ascending=asc, na_position="last")

    df = pd.DataFrame(index=df_raw.index)
    for col in df_raw.columns:
        if "PassRate" in col:
            df[col] = df_raw[col].map(lambda x: "—" if not np.isfinite(x) else f"{x:.1f}")
        elif "Avg_p_uc" in col:
            df[col] = df_raw[col].map(lambda x: "—" if not np.isfinite(x) else f"{x:.4f}")
        elif "Avg_LR_uc" in col:
            df[col] = df_raw[col].map(lambda x: "—" if not np.isfinite(x) else f"{x:.4f}")
        else:
            df[col] = df_raw[col].map(lambda x: "—" if not np.isfinite(x) else f"{x:.6f}")

    note = (
        "Cross-market averages from saved OOS risk artifacts. PassRate@level = "
        "percentage of markets where the test does not reject at 5%%. "
        "UC = Kupiec unconditional coverage (p_uc); CC = Christoffersen conditional coverage (p_cc). "
        "Global rates pool both levels (0.99 and 0.975) across all markets. "
        "Higher pass rates and p-values are preferable; lower LR statistics, Avg_VaR_t and Avg_ES indicate tighter risk forecasts."
    )
    stem = out_dir / "Table12_Risk_CrossMarket_Summary"
    _save_all(df, stem, "Table 12. Cross-Market Risk Summary", "tab:risk_cross_market", note)


# ──────────────────────────────────────────────────────────────────────────────
# Table 4e — GARCH(1,1) estimation (all four series)
# ──────────────────────────────────────────────────────────────────────────────

GARCH11_FOLDERS = ["GARCH11", "GARCH_1_1_", "GARCH(1,1)"]


def _load_garch_params(models_dir: Path, series: str) -> tuple[dict, dict]:
    for folder in GARCH11_FOLDERS:
        p = models_dir / folder / series / "params.json"
        fi = models_dir / folder / series / "fit_info.json"
        if p.exists():
            params = json.loads(p.read_text())
            fit    = json.loads(fi.read_text()) if fi.exists() else {}
            return params, fit
    return {}, {}


def build_table4_estimation(
    cfg: dict,
    models_dir: Path,
    processed_dir: Path,
    out_dir: Path,
) -> None:
    series_list = [s["name"] for s in cfg["series"]]
    rows = []

    PARAM_ROWS = [
        ("Panel A: Parameters", None),
        ("ω (omega)", "omega"),
        ("α₁ (alpha[1])", "alpha[1]"),
        ("β₁ (beta[1])", "beta[1]"),
        ("ν (nu)", "nu"),
        ("Panel B: Persistence", None),
        ("α₁+β₁", "_persistence"),
        ("Half-life (days)", "_half_life"),
        ("Stationary (α+β<1)", "_stationary"),
        ("Unconditional σ² = ω/(1−α−β)", "_uncond_var"),
        ("Panel C: In-sample fit", None),
        ("LL (in-sample)", "_LL"),
        ("AIC", "_AIC"),
        ("BIC", "_BIC"),
        ("N (obs)", "_n_obs"),
        ("Panel D: Diagnostics", None),
        ("Convergence flag", "_conv"),
        ("Hessian PD", "_hess_pd"),
    ]

    for label, key in PARAM_ROWS:
        if key is None:
            row = {s: "" for s in series_list}
            row["_label"] = label
        else:
            row = {"_label": label}
            for series in series_list:
                params, fi = _load_garch_params(models_dir, series)
                if key.startswith("_"):
                    # From fit_info
                    k2 = {
                        "_persistence": "persistence",
                        "_half_life":   "half_life_days",
                        "_stationary":  "stationary",
                        "_uncond_var":  "uncond_var",
                        "_LL":          "LL_insample",
                        "_AIC":         "AIC",
                        "_BIC":         "BIC",
                        "_n_obs":       "n_obs",
                        "_conv":        "convergence",
                        "_hess_pd":     "hess_pd",
                    }.get(key, key.lstrip("_"))
                    v = fi.get(k2, None)
                else:
                    v = params.get(key, None)

                if v is None:
                    row[series] = "—"
                elif isinstance(v, bool):
                    row[series] = str(v)
                elif isinstance(v, (int, float)) and np.isfinite(v):
                    row[series] = f"{v:.4f}"
                else:
                    row[series] = str(v)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("_label")
    df.index.name = "Parameter"

    note = (
        "GARCH(1,1) with Student-t innovations estimated by MLE. "
        "Returns in 100×log-ret (pp). "
        "Standard errors: Bollerslev–Wooldridge robust (outer-product of gradient). "
        "Unconditional variance = ω/(1−α₁−β₁); compared to empirical ε² variance "
        f"(warning if deviation > {cfg.get('unconditional_var_tolerance', 0.20):.0%}). "
        "Half-life = ln(0.5)/ln(α₁+β₁). *** p<0.01, ** p<0.05, * p<0.10."
    )
    stem = out_dir / "Table4_GARCH11_Estimation"
    _save_all(df, stem, "Table 4. GARCH(1,1) Parameter Estimates",
              "tab:garch_est", note)


# ──────────────────────────────────────────────────────────────────────────────
# Tables A1–A4 — Full estimation by series
# ──────────────────────────────────────────────────────────────────────────────

ECON_MODEL_FOLDERS = {
    "ARCH(1)":        ["ARCH1", "ARCH_1_"],
    "GARCH(1,1)":     ["GARCH11", "GARCH_1_1_"],
    "EGARCH(1,1)":    ["EGARCH11", "EGARCH_1_1_"],
    "GJR-GARCH(1,1)": ["GJR-GARCH11", "GJR-GARCH_1_1_", "GJR_GARCH_1_1_"],
    "FIGARCH(1,d,1)": ["FIGARCH1d1", "FIGARCH_1_d_1_", "FIGARCH11d"],
    "HAR":            ["HAR"],
    "MSGARCH(1,1)":   ["MSGARCH"],
}

PARAM_KEYS_BY_MODEL = {
    "ARCH(1)":        ["omega", "alpha[1]", "nu"],
    "GARCH(1,1)":     ["omega", "alpha[1]", "beta[1]", "nu"],
    "EGARCH(1,1)":    ["omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"],
    "GJR-GARCH(1,1)": ["omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"],
    "FIGARCH(1,d,1)": ["omega", "phi[1]", "d", "beta[1]", "nu"],
    "HAR":            ["const", "beta_d", "beta_w", "beta_m"],
    "MSGARCH(1,1)":   ["omega1", "alpha1", "beta1", "omega2", "alpha2", "beta2", "p11", "p22"],
}

ALL_PARAM_ROWS = [
    "omega", "alpha[1]", "gamma[1]", "beta[1]", "phi[1]", "d",
    "const", "beta_d", "beta_w", "beta_m",
    "omega1", "alpha1", "beta1", "omega2", "alpha2", "beta2", "p11", "p22",
    "nu",
    "LL_insample", "AIC", "BIC", "convergence", "hess_pd", "n_obs",
]


def _load_model_data(models_dir: Path, model_display: str, series: str) -> tuple[dict, dict]:
    for folder in ECON_MODEL_FOLDERS.get(model_display, []):
        p  = models_dir / folder / series / "params.json"
        fi = models_dir / folder / series / "fit_info.json"
        if p.exists():
            params  = json.loads(p.read_text())
            fit_inf = json.loads(fi.read_text()) if fi.exists() else {}
            return params, fit_inf
    return {}, {}


def build_table_Ax(
    series: str,
    idx: int,
    models_dir: Path,
    out_dir: Path,
) -> None:
    models = list(ECON_MODEL_FOLDERS.keys())
    rows   = []

    for param_key in ALL_PARAM_ROWS:
        row = {"_param": param_key}
        for model in models:
            params, fi = _load_model_data(models_dir, model, series)
            combined   = {**params, **fi}
            v = combined.get(param_key, None)
            if v is None:
                row[model] = "—"
            elif isinstance(v, bool):
                row[model] = str(v)
            elif isinstance(v, (int, float)) and np.isfinite(v):
                row[model] = f"{v:.4f}"
            else:
                row[model] = str(v)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("_param")
    df.index.name = "Parameter"

    note = (
        f"Series: {series}. Returns in 100×log-ret (pp). All parametric models: "
        "Student-t innovations. '—' = parameter not defined in this specification. "
        "LL_insample: total log-likelihood on training sample. "
        "Standard errors: Bollerslev–Wooldridge robust. *** p<0.01."
    )
    stem = out_dir / f"TableA{idx}_Estimation_{series}"
    _save_all(
        df, stem,
        f"Table A{idx}. Parameter Estimates — {series}",
        f"tab:est_{series.lower()}",
        note,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Table B1 — Ablation ladder (Section 7 / Proposition 2 verification)
# ──────────────────────────────────────────────────────────────────────────────

def build_table_b1(tables_dir: Path) -> None:
    """
    Table B1: per series x rung (0-3), the recovered/estimated GARCH-like
    parameters (rungs 0-1) or a cross-reference to Table 11's gate
    correspondence (rungs 2-3, where alpha/beta are not directly
    estimated parameters but gate statistics -- see
    src.eval.gate_correspondence, Section 9.3), plus implied
    persistence, half-life, QLIKE(OOS), LL_t(OOS).

    Source: outputs/tables/ablation_ladder_raw.json
    (src.models.ablation_ladder.run). Rungs 2-3 rows are left with
    alpha/beta/omega/nu = '—' if that model hasn't been trained yet, or
    if not yet cross-referenced to Table 11.
    """
    raw_path = tables_dir / "ablation_ladder_raw.json"
    if not raw_path.exists():
        log.warning("ablation_ladder_raw.json not found — skipping Table B1 "
                     "(run `python -m src.models.ablation_ladder` first).")
        return
    rows_raw = json.loads(raw_path.read_text())

    rung_labels = {
        0: "0 — GARCH(1,1)-t (MLE)",
        1: "1 — Constrained LSTM (pure Student-t MLE)",
        2: "2 — LSTM-SSE (free gates, SSE loss)",
        3: "3 — LSTM-SSE-t-Student (proposed)",
    }

    rows = []
    for r in rows_raw:
        rows.append({
            "Series": r["series"],
            "Rung": rung_labels.get(r["rung"], str(r["rung"])),
            "alpha (or E[i_t], Table 11)": _fmt(r.get("alpha"), 4),
            "beta (or E[f_t], Table 11)": _fmt(r.get("beta"), 4),
            "omega": _fmt(r.get("omega"), 4),
            "nu": _fmt(r.get("nu"), 3),
            "Persistence": _fmt(r.get("persistence"), 4),
            "Half-life (days)": _fmt(r.get("half_life_days"), 1),
            "QLIKE (OOS)": _fmt(r.get("qlike_oos"), 4),
            "LL_t (OOS)": _fmt(r.get("ll_t_oos"), 4),
            "MCS member (90%)": "—" if r.get("mcs_member") is None else str(r["mcs_member"]),
        })

    df = pd.DataFrame(rows).set_index(["Series", "Rung"])

    note = (
        "Rung 0: GARCH(1,1)-t maximum likelihood (reference). Rung 1: single-unit "
        "LSTM cell with input/forget gates held constant-but-trainable (kernels "
        "fixed at zero), output gate identically 1, initial cell state fixed, "
        "trained by pure Student-t log-likelihood (L-BFGS-B) from a neutral "
        "start (alpha0=0.05, beta0=0.85) -- NOT the GARCH answer -- so recovering "
        "alpha\\_hat, beta\\_hat is a genuine, non-circular test of Proposition 2 "
        "(see logs/proposition2\\_check.log for the PASS/FAIL verdict per "
        "series, tolerance 10\\% relative error). Rungs 2-3 report the "
        "already-trained LSTM-SSE / LSTM-SSE-t-Student models as-is; their "
        "gate statistics (analogous to alpha, beta under Proposition 2's "
        "mapping) are in Table 11, not duplicated here. Half-life = "
        "ln(0.5)/ln(beta). QLIKE and LL\\_t evaluated OOS on the test split."
    )
    stem = tables_dir / "TableB1_ablation_ladder"
    _save_all(df, stem, "Table B1. Ablation Ladder — Proposition 2 Verification",
              "tab:ablation_b1", note)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg           = load_config(config_path)
    tables_dir    = Path(cfg["paths"]["tables"])
    models_dir    = Path(cfg["paths"]["models"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_path = tables_dir / "raw_results.json"
    if not raw_path.exists():
        log.error("raw_results.json not found — run eval step first.")
        return
    all_results = json.loads(raw_path.read_text())

    enc_path = tables_dir / "encompassing_results.json"
    encompassing_all = json.loads(enc_path.read_text()) if enc_path.exists() else {}

    series_list = [s["name"] for s in cfg["series"]]

    # ── Table 3: roster ───────────────────────────────────────────────────────
    log.info("Building Table 3 (roster) …")
    build_table3(tables_dir)

    # ── Tables 4–7: OOS performance ───────────────────────────────────────────
    for i, series in enumerate(series_list, start=4):
        log.info("Building Table %d (OOS %s) …", i, series)
        build_table_oos(series, all_results.get(series, {}), tnum=i, out_dir=tables_dir)

    # ── Table 8: cross-market summary ─────────────────────────────────────────
    log.info("Building Table 8 (cross-market) …")
    build_table8(all_results, series_list, tables_dir)

    # ── Table 9: Diebold–Mariano ──────────────────────────────────────────────
    log.info("Building Table 9 (DM) …")
    build_table9(all_results, series_list, tables_dir)

    # ── Table 10: Forecast-Encompassing ───────────────────────────────────────
    if encompassing_all:
        log.info("Building Table 10 (Encompassing) …")
        build_table10(encompassing_all, series_list, tables_dir)
    else:
        log.warning("encompassing_results.json not found — skipping Table 10.")

    # ── Table 11a–11d: Risk backtests ───────────────────────────────────────
    for tlabel, series in zip(["a", "b", "c", "d"], series_list):
        log.info("Building Table 11%s (Risk %s) …", tlabel, series)
        build_table11_risk(series, all_results.get(series, {}), tlabel=tlabel, out_dir=tables_dir)

    # ── Table 12: cross-market risk summary ──────────────────────────────────
    log.info("Building Table 12 (Risk cross-market summary) …")
    build_table12_risk_summary(all_results, series_list, tables_dir)

    # ── Table 4e: GARCH estimation ────────────────────────────────────────────
    log.info("Building Table 4e (GARCH estimation) …")
    build_table4_estimation(cfg, models_dir, processed_dir, tables_dir)

    # ── Tables A1–A4: full estimation ─────────────────────────────────────────
    for idx, series in enumerate(series_list, start=1):
        log.info("Building Table A%d (estimation %s) …", idx, series)
        build_table_Ax(series, idx, models_dir, tables_dir)

    # ── Table B1: ablation ladder (Section 7 / Proposition 2) ────────────────
    log.info("Building Table B1 (ablation ladder) …")
    build_table_b1(tables_dir)

    log.info("══ build_tables complete — outputs in %s ══", tables_dir)


def main():
    parser = argparse.ArgumentParser(description="Build all paper tables")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    import argparse
    main()
