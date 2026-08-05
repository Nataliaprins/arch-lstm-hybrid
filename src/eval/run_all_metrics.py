"""
run_all_metrics.py — Orchestrates all OOS evaluations and writes raw_results.json.

Usage:
    python -m src.eval.run_all_metrics --config config/config.yaml

Output
------
outputs/tables/raw_results.json  — nested dict:
    {series: {model: {metrics, dm_qlike, dm_mse, mcs, bootstrap, var_es}}}

This JSON is consumed by build_tables.py and build_figures.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics     import compute_all, qlike_array, mse_array
from src.eval.dm_test     import run_dm_battery, run_tost_battery
from src.eval.mcs         import mcs
from src.eval.bootstrap   import bootstrap_ci_all
from src.eval.var_es_backtest import run_backtest
from src.eval.encompassing_test import run_encompassing_battery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# nu used for LL_t and VaR (default; per-model nu stored in best_hparams.json)
DEFAULT_NU = 5.0

# Section 9.7: positive variance floor, as a fraction of the training-split
# unconditional variance -- applied to every model's predictions before any
# metric is computed (see the diagnostic that named SVR-GARCH, CNN-LSTM/ETH,
# and NN-GARCH/SP500 specifically; applied uniformly here since a near-zero
# variance forecast is an equally-bad numerical hazard from any model).
VARIANCE_FLOOR_FACTOR = 1e-6

# A model is retired from that series' tables (Section 9.7) if its QLIKE,
# even AFTER the floor above, is still this many times the median QLIKE of
# every other model in the same series -- i.e. the floor alone could not
# fix it, so the model is genuinely broken there, not just numerically
# fragile. Documented in logs/broken_models.log, never silently dropped.
ABERRANT_QLIKE_MEDIAN_MULTIPLE = 20.0
ABERRANT_QLIKE_ABS_FLOOR = 50.0  # also require QLIKE to be implausibly large in absolute terms

# ── Forecast-encompassing pairs (candidate, benchmark) ──────────────────────
# LSTM-SSE-t-Student is checked against both ARCH(1) (the model it is meant
# to echo — heavy-tailed, first-order recursion) and GARCH(1,1) (the
# project's canonical benchmark). NN-GARCH is checked against GARCH(1,1)
# since its input is literally the GARCH(1,1) filtered variance.
ENCOMPASSING_PAIRS = [
    ("LSTM-SSE-t-Student", "ARCH(1)"),
    ("LSTM-SSE-t-Student", "GARCH(1,1)"),
    ("NN-GARCH",           "GARCH(1,1)"),
]


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _read_eps(processed_dir: Path, series: str, split: str) -> np.ndarray:
    f = processed_dir / series / f"{split}_eps.csv"
    return pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].values.astype(float)


# ── Model discovery ──────────────────────────────────────────────────────────

# Canonical display names (folder_name → display_name)
ECON_FOLDERS = {
    "ARCH1":           "ARCH(1)",
    "ARCH_1_":         "ARCH(1)",
    "GARCH11":         "GARCH(1,1)",
    "GARCH_1_1_":      "GARCH(1,1)",
    "EGARCH11":        "EGARCH(1,1)",
    "EGARCH_1_1_":     "EGARCH(1,1)",
    "GJR-GARCH11":     "GJR-GARCH(1,1)",
    "GJR-GARCH_1_1_":  "GJR-GARCH(1,1)",
    "GJR_GARCH_1_1_":  "GJR-GARCH(1,1)",
    "FIGARCH1d1":      "FIGARCH(1,d,1)",
    "FIGARCH_1_d_1_":  "FIGARCH(1,d,1)",
    "FIGARCH11d":      "FIGARCH(1,d,1)",
    "HAR":             "HAR",
    "MSGARCH":         "MSGARCH(1,1)",
}
NEURAL_FOLDERS = {
    "SVR-GARCH":         "SVR-GARCH",
    "LSTM-SSE":          "LSTM-SSE",
    "NN-GARCH":          "NN-GARCH",
    "CNN-LSTM":          "CNN-LSTM",
    "LSTM-Attention":    "LSTM-Attention",
    "TCN":               "TCN",
    "Transformer":       "Transformer",
    "LSTM-SSE-t-Student": "LSTM-SSE-t-Student",
    "ARCH-LSTM":         "ARCH-LSTM",
}

# Section 9.2: minimum-bar reference forecast, fixed at the training-split
# unconditional variance for the whole OOS window. Injected into sigma2_all
# like any other model so it automatically gets the same metrics/DM/MCS
# treatment as every real model (Tables 4-7, 8, 9).
CONSTANT_MODEL_NAME = "Constant (unconditional variance)"


def _load_sigma2(model_dir: Path, series: str) -> np.ndarray | None:
    npy_path = model_dir / series / "sigma2_test.npy"
    if npy_path.exists():
        return np.load(npy_path)
    # Section 9.7: R/msgarch.R saves sigma2_test.csv (a "sigma2_test"
    # column), not .npy -- MSGARCH never appeared in any results table
    # because this loader only ever looked for the .npy file, silently
    # ignoring the CSV the R script had been producing all along.
    csv_path = model_dir / series / "sigma2_test.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)["sigma2_test"].values.astype(float)
    return None


def _load_sigma2_per_seed(model_dir: Path, series: str) -> np.ndarray | None:
    p = model_dir / series / "sigma2_per_seed.npy"
    if p.exists():
        return np.load(p)
    return None


def _load_nu(model_dir: Path, series: str) -> float:
    p = model_dir / series / "best_hparams.json"
    if p.exists():
        hp = json.loads(p.read_text())
        return float(hp.get("nu", DEFAULT_NU))
    return DEFAULT_NU


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg           = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_data"])
    models_dir    = Path(cfg["paths"]["models"])
    tables_dir    = Path(cfg["paths"]["tables"])
    logs_dir      = Path(cfg["paths"]["logs"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "broken_models.log").write_text("")  # Section 9.7: fresh per run, not appended across runs

    mcs_cfg  = cfg.get("mcs", {})
    mcs_B    = mcs_cfg.get("n_bootstrap", 10_000)
    mcs_blk  = mcs_cfg.get("block_size", 20)
    mcs_lvl  = mcs_cfg.get("level", 0.90)
    var_lvls = cfg.get("var_confidence_levels", [0.99, 0.975])
    tost_delta_pct = cfg.get("tost", {}).get("delta_pct", 0.02)

    all_results: dict = {}
    encompassing_all: dict = {}

    for sc in cfg["series"]:
        series = sc["name"]
        log.info("══ Evaluating: %s ════════════════════════════════════════", series)

        test_eps  = _read_eps(processed_dir, series, "test")
        test_eps2 = test_eps ** 2

        # ── Collect all sigma2_hat arrays ────────────────────────────────────
        sigma2_all:      dict[str, np.ndarray] = {}
        sigma2_per_seed: dict[str, np.ndarray] = {}
        nu_by_model:     dict[str, float]      = {}

        # Section 9.2: constant reference forecast, fixed at the training-split
        # unconditional variance -- deliberately independent of Section 4's
        # input scaler choice, computed directly from train_eps2.
        train_eps2 = _read_eps(processed_dir, series, "train") ** 2
        sigma2_train_uncond = float(np.mean(train_eps2))
        sigma2_all[CONSTANT_MODEL_NAME]  = np.full(len(test_eps2), sigma2_train_uncond)
        nu_by_model[CONSTANT_MODEL_NAME] = DEFAULT_NU

        # Section 9.7: positive variance floor applied to every model's
        # predictions before any metric is computed -- guards QLIKE/LL_t/
        # VaR against exploding when a model (SVR-GARCH in particular,
        # also observed for CNN-LSTM/ETH and NN-GARCH/SP500) predicts a
        # near-zero variance for a handful of observations.
        floor = VARIANCE_FLOOR_FACTOR * sigma2_train_uncond

        # Econometric models
        for folder, display in {**ECON_FOLDERS}.items():
            model_dir = models_dir / folder
            if not model_dir.exists():
                continue
            s2 = _load_sigma2(model_dir, series)
            if s2 is not None and len(s2) > 0:
                # Align lengths
                n = min(len(s2), len(test_eps2))
                sigma2_all[display]  = np.maximum(s2[:n], floor)
                nu_by_model[display] = DEFAULT_NU
                log.info("  [%s] loaded %s: %d obs", series, display, n)

        # Neural models
        for folder, display in NEURAL_FOLDERS.items():
            model_dir = models_dir / folder
            if not model_dir.exists():
                continue
            s2 = _load_sigma2(model_dir, series)
            if s2 is not None and len(s2) > 0:
                n = min(len(s2), len(test_eps2))
                sigma2_all[display]  = np.maximum(s2[:n], floor)
                nu_by_model[display] = _load_nu(model_dir, series)
                per_seed = _load_sigma2_per_seed(model_dir, series)
                if per_seed is not None:
                    sigma2_per_seed[display] = np.maximum(per_seed[:, :n], floor)
                log.info("  [%s] loaded %s: %d obs", series, display, n)

        if "GARCH(1,1)" not in sigma2_all:
            log.warning("[%s] GARCH(1,1) not found — Δ%% will be NaN", series)
        garch_s2 = sigma2_all.get("GARCH(1,1)", None)

        # ── Forecast-encompassing test ───────────────────────────────────────
        enc_result = run_encompassing_battery(test_eps2, sigma2_all, ENCOMPASSING_PAIRS)
        encompassing_all[series] = enc_result
        for pair_key, res in enc_result.items():
            if "error" in res:
                log.warning("  [%s] Encompassing %s: %s", series, pair_key, res["error"])
            else:
                log.info(
                    "  [%s] Encompassing %-40s verdict=%-22s",
                    series, pair_key, res["verdict"],
                )
                log.info(
                    "      beta_bench=%.4f (p=%.4f)  beta_cand=%.4f (p=%.4f)",
                    res["beta_benchmark"], res["p_benchmark"],
                    res["beta_candidate"], res["p_candidate"],
                )

        # ── Per-model metrics ────────────────────────────────────────────────
        model_results: dict = {}
        qlike_arrays:  dict[str, np.ndarray] = {}
        mse_arrays:    dict[str, np.ndarray] = {}

        for model, s2 in sigma2_all.items():
            n    = len(s2)
            e2   = test_eps2[:n]
            eps  = test_eps[:n]
            nu   = nu_by_model.get(model, DEFAULT_NU)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = compute_all(
                    s2, e2,
                    sigma2_hat_benchmark=garch_s2[:n] if garch_s2 is not None else None,
                    nu=nu,
                )

            # Multi-seed mean ± std. Section 9.7: a model with exactly one
            # seed (SVR-GARCH, deterministic by construction) has no
            # seed-to-seed variation to measure -- ddof=1 std of a single
            # observation is NaN, not "very small", so it must be flagged
            # explicitly rather than rendered as "± nan".
            std_dict = {}
            if model in sigma2_per_seed:
                per = sigma2_per_seed[model]  # (S, n)
                if per.shape[0] < 2:
                    std_dict = {"deterministic": True}
                else:
                    seeds_mse = [
                        float(np.mean((per[i] - e2)**2)) for i in range(per.shape[0])
                    ]
                    seeds_mae = [
                        float(np.mean(np.abs(per[i] - e2))) for i in range(per.shape[0])
                    ]
                    std_dict = {
                        "MSE_std":  round(float(np.std(seeds_mse, ddof=1)), 6),
                        "MAE_std":  round(float(np.std(seeds_mae, ddof=1)), 6),
                    }

            # Bootstrap CI
            try:
                boot = bootstrap_ci_all(s2, e2, B=2000, block_size=20, seed=42)
            except Exception:
                boot = {}

            # VaR/ES backtest
            try:
                var_es = run_backtest(eps, s2, nu=nu, levels=var_lvls)
            except Exception:
                var_es = {}

            model_results[model] = {
                "metrics":    {k: (round(v, 6) if np.isfinite(v) else None) for k, v in metrics.items()},
                "std":        std_dict,
                "bootstrap":  {k: round(v, 6) for k, v in boot.items() if np.isfinite(v)},
                "var_es":     var_es,
                "n_test":     n,
                "nu":         nu,
            }

            # Loss arrays for DM / MCS
            qlike_arrays[model] = qlike_array(s2, e2)
            mse_arrays[model]   = mse_array(s2, e2)

            log.info(
                "  [%s/%s] MSE=%.5f  RMSE=%.5f  MAE=%.5f  R2=%.4f  QLIKE=%.5f",
                series, model,
                metrics["MSE"], metrics["RMSE"], metrics["MAE"],
                metrics["R2"],  metrics["QLIKE"],
            )

        # ── Retire models still aberrant after the variance floor (Section 9.7) ──
        # Never silently drop -- log exactly what was removed and why.
        finite_qlikes = {
            m: r["metrics"]["QLIKE"] for m, r in model_results.items()
            if r["metrics"].get("QLIKE") is not None and np.isfinite(r["metrics"]["QLIKE"])
        }
        if len(finite_qlikes) >= 2:
            median_qlike = float(np.median(list(finite_qlikes.values())))
            broken_this_series = []
            for model, q in list(finite_qlikes.items()):
                is_aberrant = (
                    q > ABERRANT_QLIKE_ABS_FLOOR
                    and q > ABERRANT_QLIKE_MEDIAN_MULTIPLE * median_qlike
                )
                if is_aberrant:
                    broken_this_series.append((model, q, median_qlike))
            for model, q, med in broken_this_series:
                msg = (
                    f"[{series}] {model} retired: QLIKE={q:.4f} even after the "
                    f"{VARIANCE_FLOOR_FACTOR:.0e}xσ̄²_train variance floor "
                    f"(> {ABERRANT_QLIKE_MEDIAN_MULTIPLE:.0f}x the series median "
                    f"of {med:.4f} and > absolute floor {ABERRANT_QLIKE_ABS_FLOOR:.0f}) "
                    "-- excluded from Tables 4-7/8/9/MCS for this series."
                )
                log.error(msg)
                with open(logs_dir / "broken_models.log", "a") as fh:
                    fh.write(msg + "\n")
                del model_results[model]
                qlike_arrays.pop(model, None)
                mse_arrays.pop(model, None)
                sigma2_per_seed.pop(model, None)

        # ── MCS (Table 4–7 last column) ──────────────────────────────────────
        if len(qlike_arrays) >= 2:
            names = list(qlike_arrays)
            n_min = min(len(v) for v in qlike_arrays.values())
            loss_mat = np.column_stack([qlike_arrays[m][:n_min] for m in names])
            try:
                mcs_result = mcs(
                    loss_mat, names,
                    alpha=1 - mcs_lvl, B=mcs_B, block_size=mcs_blk,
                    seed=cfg["seed"],
                )
            except Exception as exc:
                log.warning("[%s] MCS failed: %s", series, exc)
                mcs_result = {m: None for m in names}
        else:
            mcs_result = {m: None for m in sigma2_all}

        for model in model_results:
            model_results[model]["mcs_90"] = mcs_result.get(model, None)

        # ── Diebold–Mariano (LSTM-SSE-t-Student vs. all) ─────────────────────
        proposed = "LSTM-SSE-t-Student"
        if proposed in qlike_arrays:
            proposed_ql  = qlike_arrays[proposed]
            proposed_mse = mse_arrays[proposed]
            rivals_ql    = {k: v for k, v in qlike_arrays.items() if k != proposed}
            rivals_mse   = {k: v for k, v in mse_arrays.items()   if k != proposed}

            # Align lengths
            n_min = min(len(proposed_ql), min((len(v) for v in rivals_ql.values()), default=len(proposed_ql)))
            prop_ql  = proposed_ql[:n_min]
            prop_mse = proposed_mse[:n_min]
            riv_ql   = {k: v[:n_min] for k, v in rivals_ql.items()}
            riv_mse  = {k: v[:n_min] for k, v in rivals_mse.items()}

            try:
                dm_qlike = run_dm_battery(prop_ql, riv_ql, loss_type="QLIKE")
            except Exception as exc:
                log.warning("[%s] DM QLIKE failed: %s", series, exc)
                dm_qlike = {}
            try:
                dm_mse_res = run_dm_battery(prop_mse, riv_mse, loss_type="MSE")
            except Exception as exc:
                log.warning("[%s] DM MSE failed: %s", series, exc)
                dm_mse_res = {}

            # ── TOST equivalence (Section 9.4): margin = tost_delta_pct × GARCH(1,1)'s own mean QLIKE ──
            tost_result: dict = {}
            if "GARCH(1,1)" in qlike_arrays:
                delta = tost_delta_pct * float(np.mean(qlike_arrays["GARCH(1,1)"]))
                try:
                    tost_result = run_tost_battery(prop_ql, riv_ql, delta=delta)
                except Exception as exc:
                    log.warning("[%s] TOST failed: %s", series, exc)
            else:
                log.warning("[%s] GARCH(1,1) not found — TOST skipped (margin is undefined)", series)

            for model in model_results:
                model_results[model]["dm_qlike"] = dm_qlike.get(model, {})
                model_results[model]["dm_mse"]   = dm_mse_res.get(model, {})
                model_results[model]["tost"]     = tost_result.get(model, {})
        else:
            log.warning("[%s] Proposed model '%s' not found — DM/TOST skipped", series, proposed)

        all_results[series] = model_results

    # ── Verify unconditional variance (req. 7 / Table 4 note) ────────────────
    tol = cfg.get("unconditional_var_tolerance", 0.20)
    for sc in cfg["series"]:
        series  = sc["name"]
        meta_f  = processed_dir / series / "meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        emp_var = meta.get("train_var_eps", float("nan"))

        garch_dir = models_dir / "GARCH11" / series
        if not garch_dir.exists():
            garch_dir = models_dir / "GARCH_1_1_" / series
        fi_path = garch_dir / "fit_info.json"
        if fi_path.exists():
            fi = json.loads(fi_path.read_text())
            uncond = fi.get("uncond_var", None)
            if uncond and np.isfinite(uncond) and emp_var > 0:
                rel_diff = abs(uncond - emp_var) / emp_var
                msg = (
                    f"[{series}] GARCH(1,1) unconditional var={uncond:.4f}  "
                    f"empirical ε² variance={emp_var:.4f}  "
                    f"relative diff={rel_diff:.1%}"
                )
                if rel_diff > tol:
                    log.warning("WARN: %s  (> tolerance %.0f%%)", msg, tol * 100)
                    with open(logs_dir / f"uncond_var_check_{series}.log", "w") as fh:
                        fh.write("WARNING: " + msg + "\n")
                else:
                    log.info("OK: %s", msg)

    # ── Save raw results ──────────────────────────────────────────────────────
    out_path = tables_dir / "raw_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    log.info("Raw results saved to %s", out_path)

    # ── Save encompassing-test results (separate file — not a model roster) ──
    enc_path = tables_dir / "encompassing_results.json"
    with open(enc_path, "w") as fh:
        json.dump(encompassing_all, fh, indent=2, default=str)
    log.info("Encompassing-test results saved to %s", enc_path)

    log.info("══ run_all_metrics complete ══════════════════════════════════")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
