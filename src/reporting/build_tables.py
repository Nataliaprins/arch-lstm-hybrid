"""
build_tables.py — Emit Tables 3–9 and 4+A1–A4 in .csv, .tex (booktabs), .docx.

Usage:
    python -m src.reporting.build_tables --config config/config.yaml

Input:  outputs/tables/raw_results.json
        outputs/models/<model>/<series>/{params.json, fit_info.json}
Output: outputs/tables/  *.csv  *.tex  *.docx

Table numbering (paper convention)
-----------------------------------
Table 3   — Model roster (static)
Tables 4–7 — OOS performance, one per series
Table 8   — Cross-market Δ% summary
Table 9   — Diebold–Mariano
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

    # ── Table 4e: GARCH estimation ────────────────────────────────────────────
    log.info("Building Table 4e (GARCH estimation) …")
    build_table4_estimation(cfg, models_dir, processed_dir, tables_dir)

    # ── Tables A1–A4: full estimation ─────────────────────────────────────────
    for idx, series in enumerate(series_list, start=1):
        log.info("Building Table A%d (estimation %s) …", idx, series)
        build_table_Ax(series, idx, models_dir, tables_dir)

    log.info("══ build_tables complete — outputs in %s ══", tables_dir)


def main():
    parser = argparse.ArgumentParser(description="Build all paper tables")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    import argparse
    main()
