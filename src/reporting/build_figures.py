"""
build_figures.py — Emit figures for the paper.

Figures produced
----------------
1. lambda_sensitivity_<series>.pdf   — Δ%MSE vs λ, one panel per series
2. trainval_curves_<series>_<model>.pdf — train/val loss curves (all seeds)
3. gate_dynamics_<series>.pdf        — mean ± std of per-seed σ²_test over time
4. var_backtest_<series>.pdf         — VaR hit sequences
5. forecast_vs_observed_<series>.png — σ̂²_t (mean over seeds) vs ε²_t across
                                       train / validation / test(OOS)

Usage:
    python -m src.reporting.build_figures --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":     "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

COLORS = plt.cm.tab10.colors


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _savefig(fig, path: Path) -> None:
    fig.savefig(str(path))
    plt.close(fig)
    log.info("Saved figure: %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — λ sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def build_lambda_sensitivity(series_list: list[str], models_dir: Path, fig_dir: Path) -> None:
    """Build the λ-sensitivity figure."""
    candidates = [
        models_dir / "lambda_sensitivity",
        models_dir / "LSTM-SSE-t-Student" / "lambda_sensitivity",
        models_dir / "LSTM-SSE-t-Student",
    ]
    lam_dir = next((p for p in candidates if p.exists()), None)
    if lam_dir is None:
        log.warning("lambda_sensitivity dir not found in any expected location; skipping.")
        return

    fig, axes = plt.subplots(1, len(series_list), figsize=(4 * len(series_list), 4), sharey=False)
    if len(series_list) == 1:
        axes = [axes]

    for ax, series in zip(axes, series_list):
        jf = lam_dir / f"{series}_lambda_sensitivity.json"
        if not jf.exists():
            alt = lam_dir / series / f"{series}_lambda_sensitivity.json"
            if alt.exists():
                jf = alt
            else:
                ax.set_title(f"{series} (no data)")
                continue

        data = json.loads(jf.read_text())
        lambdas = [d.get("lambda") for d in data]
        mses    = [d.get("mse")    for d in data]

        ax.plot(lambdas, mses, "o-", color=COLORS[0], linewidth=2, markersize=6)
        ax.axvline(lambdas[int(np.argmin([m for m in mses if np.isfinite(m)] or [0]))],
                   color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("λ (mixing weight)")
        ax.set_ylabel("OOS MSE")
        ax.set_title(series)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

    fig.suptitle("λ-Sensitivity Analysis — LSTM-SSE-t-Student", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, fig_dir / "lambda_sensitivity.pdf")


def build_trainval_curves(
    series_list: list[str],
    models_dir:  Path,
    fig_dir:     Path,
    model_folder: str = "LSTM-SSE-t-Student",
) -> None:
    for series in series_list:
        hist_f = models_dir / model_folder / series / "histories.json"
        if not hist_f.exists():
            log.warning("histories.json not found for %s/%s; skipping.", model_folder, series)
            continue

        histories = json.loads(hist_f.read_text())
        S = len(histories)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, hist in enumerate(histories):
            tr = hist.get("train_loss", [])
            vl = hist.get("val_loss",   [])
            c  = COLORS[i % len(COLORS)]
            axes[0].plot(tr, color=c, alpha=0.6, linewidth=1)
            axes[1].plot(vl, color=c, alpha=0.6, linewidth=1, label=f"Seed {i}")

        # Mean
        max_ep = max(len(h.get("train_loss", [])) for h in histories)
        def _pad(lst, L):
            arr = np.full(L, np.nan)
            arr[:len(lst)] = lst
            return arr

        tr_mat = np.array([_pad(h.get("train_loss", []), max_ep) for h in histories])
        vl_mat = np.array([_pad(h.get("val_loss",   []), max_ep) for h in histories])
        ep = np.arange(max_ep)

        axes[0].plot(ep, np.nanmean(tr_mat, axis=0), "k-", linewidth=2, label="Mean")
        axes[1].plot(ep, np.nanmean(vl_mat, axis=0), "k-", linewidth=2, label="Mean")

        for ax, ttl in zip(axes, ["Train loss", "Validation loss"]):
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(ttl)

        axes[1].legend(fontsize=7, ncol=2, loc="upper right")
        fig.suptitle(f"{model_folder} — {series} (S={S} seeds)", fontsize=12)
        fig.tight_layout()
        _savefig(fig, fig_dir / f"trainval_{series}_{model_folder}.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Gate dynamics / σ²_t multi-seed stability
# ──────────────────────────────────────────────────────────────────────────────

def build_gate_dynamics(
    series_list: list[str],
    models_dir:  Path,
    processed_dir: Path,
    fig_dir:     Path,
    model_folder: str = "LSTM-SSE-t-Student",
) -> None:
    for series in series_list:
        per_seed_f = models_dir / model_folder / series / "sigma2_per_seed.npy"
        if not per_seed_f.exists():
            continue

        per_seed  = np.load(per_seed_f)            # (S, n_test)
        mean_s2   = per_seed.mean(axis=0)
        std_s2    = per_seed.std(axis=0)

        # Load proxy
        proxy_f = processed_dir / series / "test_eps2.csv"
        if proxy_f.exists():
            proxy = pd.read_csv(proxy_f, index_col=0, parse_dates=True).iloc[:, 0].values
        else:
            proxy = None

        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.arange(len(mean_s2))
        ax.fill_between(t, mean_s2 - std_s2, mean_s2 + std_s2,
                        alpha=0.3, color=COLORS[0], label="Mean ± 1 s.d.")
        ax.plot(t, mean_s2, color=COLORS[0], linewidth=1.5, label="σ²_t (mean)")
        if proxy is not None:
            n = min(len(proxy), len(t))
            ax.scatter(t[:n], proxy[:n], s=2, color="gray", alpha=0.3, label="ε²_t (proxy)")
        ax.set_xlabel("OOS observation")
        ax.set_ylabel("Conditional variance")
        ax.set_title(f"{model_folder} — {series}: Multi-seed σ²_t stability")
        ax.legend(fontsize=9)
        fig.tight_layout()
        _savefig(fig, fig_dir / f"gate_dynamics_{series}.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — VaR hit sequences (proposed model)
# ──────────────────────────────────────────────────────────────────────────────

def build_var_figures(
    series_list:  list[str],
    raw_results:  dict,
    models_dir:   Path,
    processed_dir: Path,
    fig_dir:      Path,
    model:        str = "LSTM-SSE-t-Student",
    level:        float = 0.99,
) -> None:
    from src.eval.var_es_backtest import compute_var_t

    for series in series_list:
        s2_f = models_dir / model / series / "sigma2_test.npy"
        if not s2_f.exists():
            continue
        sigma2 = np.load(s2_f)

        eps_f = processed_dir / series / "test_eps.csv"
        if not eps_f.exists():
            continue
        eps = pd.read_csv(eps_f, index_col=0, parse_dates=True).iloc[:, 0].values

        n     = min(len(sigma2), len(eps))
        sigma2, eps = sigma2[:n], eps[:n]

        nu   = raw_results.get(series, {}).get(model, {}).get("nu", 5.0)
        alpha = 1.0 - level
        var   = compute_var_t(sigma2, nu=nu, alpha=alpha)
        hits  = (eps < -var).astype(float)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t = np.arange(n)
        axes[0].plot(t, eps,  color="steelblue",  linewidth=0.6, label="ε_t")
        axes[0].plot(t, -var, color="firebrick",  linewidth=1.2, linestyle="--",
                    label=f"−VaR_{level:.0%} (Student-t)")
        axes[0].scatter(t[hits == 1], eps[hits == 1], color="red", s=20, zorder=5, label="Hit")
        axes[0].set_ylabel("Return (100×log-ret)")
        axes[0].legend(fontsize=8)

        axes[1].bar(t, hits, color="firebrick", width=1, alpha=0.7)
        axes[1].set_ylabel("VaR hit")
        axes[1].set_xlabel("OOS observation")

        fig.suptitle(f"{model} — {series}  VaR {level:.0%} Hit Sequence", fontsize=12)
        fig.tight_layout()
        _savefig(fig, fig_dir / f"var_hits_{series}_{level:.0%}.pdf")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — forecast vs observed, full timeline (train / val / test)
# ──────────────────────────────────────────────────────────────────────────────

def build_forecast_vs_observed(
    series_list:   list[str],
    cfg:           dict,
    models_dir:    Path,
    processed_dir: Path,
    fig_dir:       Path,
    model_folder:  str = "LSTM-SSE-t-Student",
) -> None:
    """
    σ̂²_t (mean across seeds) vs the realized-variance proxy ε²_t, over the
    FULL timeline (train + validation + test/OOS), with the three splits
    shaded. Only OOS sigma2_test.npy is saved by the training pipeline
    (_multiseed_train_and_predict) -- train/val forecasts don't exist on
    disk anywhere, so they're rebuilt here from each seed's saved
    weights.weights.h5 (same rebuild-and-load pattern as
    src.eval.gate_correspondence.compute_gate_statistics_one_seed) and
    averaged across seeds exactly like the OOS aggregation already does.

    Window/target alignment (see src.tuning.tune_and_train._make_windows
    and _run_model_series): y_train's dates are train dates[window:]  (the
    first `window` observations have no full lookback yet); y_val's and
    the test set's dates line up 1:1 with the full val/test date ranges
    (their windows are allowed to look back across the split boundary,
    which is correct/expected sliding-window behavior, not a leak of
    labels).
    """
    import tensorflow as tf
    from src.tuning.tune_and_train import _load_series, _make_windows
    from src.models.neural import build_lstm_t_student
    from src.losses.hybrid_student_t import sigma2_from_direct_output

    W = cfg["data"]["window"]

    for series in series_list:
        model_dir = models_dir / model_folder / series
        hp_path = model_dir / "best_hparams.json"
        seed_dirs = sorted(model_dir.glob("seed_*")) if model_dir.exists() else []
        if not hp_path.exists() or not seed_dirs:
            log.warning("[%s] no best_hparams.json / seed weights under %s; skipping.", series, model_dir)
            continue
        hp = json.loads(hp_path.read_text())

        data = _load_series(series, processed_dir)
        train_x, val_x, test_x = data["train_x_scaled"], data["val_x_scaled"], data["test_x_scaled"]
        train_eps2, val_eps2, test_eps2 = data["train_eps2"], data["val_eps2"], data["test_eps2"]

        X_train, y_train = _make_windows(train_x, train_eps2, W)
        tv_x = np.concatenate([train_x, val_x])
        tv_eps2 = np.concatenate([train_eps2, val_eps2])
        X_tv, y_tv = _make_windows(tv_x, tv_eps2, W)
        X_val, y_val = X_tv[len(X_train):], y_tv[len(y_train):]

        full_x = np.concatenate([train_x, val_x, test_x])
        full_eps2 = np.concatenate([train_eps2, val_eps2, test_eps2])
        X_full, _ = _make_windows(full_x, full_eps2, W)
        n_tv = len(train_x) + len(val_x)
        X_test_w = X_full[n_tv - W:][:len(test_x)]

        sigma2_train_seeds, sigma2_val_seeds, sigma2_test_seeds = [], [], []
        try:
            for seed_dir in seed_dirs:
                weights_path = seed_dir / "weights.weights.h5"
                if not weights_path.exists():
                    continue
                model = build_lstm_t_student(hp)
                if not model.built:
                    model(X_train[:1])
                model.load_weights(str(weights_path))
                sigma2_train_seeds.append(sigma2_from_direct_output(model.predict(X_train, verbose=0).ravel()))
                sigma2_val_seeds.append(sigma2_from_direct_output(model.predict(X_val, verbose=0).ravel()))
                sigma2_test_seeds.append(sigma2_from_direct_output(model.predict(X_test_w, verbose=0).ravel()))
                tf.keras.backend.clear_session()
        except (ValueError, tf.errors.InvalidArgumentError) as exc:
            # Saved weights were trained under a different data.window than
            # the one currently in the config (e.g. a stale run predating a
            # window change) -- architecturally incompatible, not a bug in
            # this function. Skip the series instead of aborting every
            # figure after it.
            log.warning(
                "[%s] saved weights incompatible with current data.window=%d "
                "(likely trained under a different window); skipping. (%s)",
                series, W, exc,
            )
            continue

        if not sigma2_train_seeds:
            log.warning("[%s] no loadable seed weights; skipping.", series)
            continue

        sigma2_train = np.mean(sigma2_train_seeds, axis=0)
        sigma2_val   = np.mean(sigma2_val_seeds, axis=0)
        sigma2_test  = np.mean(sigma2_test_seeds, axis=0)

        dates_train_full = pd.read_csv(processed_dir / series / "train_eps2.csv", index_col=0, parse_dates=True).index
        dates_val   = pd.read_csv(processed_dir / series / "val_eps2.csv",   index_col=0, parse_dates=True).index
        dates_test  = pd.read_csv(processed_dir / series / "test_eps2.csv", index_col=0, parse_dates=True).index
        dates_train = dates_train_full[W:]

        dates    = dates_train.append(dates_val).append(dates_test)
        observed = np.concatenate([y_train, val_eps2, test_eps2])
        forecast = np.concatenate([sigma2_train, sigma2_val, sigma2_test])

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dates, observed, color="gray", linewidth=0.5, alpha=0.6, label="ε²_t (observado)")
        ax.plot(dates, forecast, color=COLORS[0], linewidth=1.1, label="σ̂²_t (pronóstico, media entre semillas)")

        val_start, test_start = dates_val[0], dates_test[0]
        ax.axvspan(dates[0], val_start, color=COLORS[2], alpha=0.06, label="Train")
        ax.axvspan(val_start, test_start, color=COLORS[1], alpha=0.10, label="Validation")
        ax.axvspan(test_start, dates[-1], color=COLORS[3], alpha=0.10, label="Test (OOS)")
        ax.axvline(val_start, color="black", linestyle="--", linewidth=0.7)
        ax.axvline(test_start, color="black", linestyle="-", linewidth=0.7)

        ax.set_yscale("log")
        ax.set_ylabel("Varianza condicional (pp², escala log)")
        ax.set_title(f"{model_folder} — {series}: pronóstico vs. observado (train / validation / test-OOS)")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        fig.tight_layout()
        _savefig(fig, fig_dir / f"forecast_vs_observed_{series}.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg           = load_config(config_path)
    figures_dir   = Path(cfg["paths"]["figures"])
    models_dir    = Path(cfg["paths"]["models"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    tables_dir    = Path(cfg["paths"]["tables"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    series_list = [s["name"] for s in cfg["series"]]

    # Load raw results (for VaR nu)
    raw_f = tables_dir / "raw_results.json"
    raw_results = json.loads(raw_f.read_text()) if raw_f.exists() else {}

    log.info("Building λ-sensitivity figure …")
    build_lambda_sensitivity(series_list, models_dir, figures_dir)

    log.info("Building train/val curves …")
    build_trainval_curves(series_list, models_dir, figures_dir)

    log.info("Building gate-dynamics figures …")
    build_gate_dynamics(series_list, models_dir, processed_dir, figures_dir)

    log.info("Building VaR hit-sequence figures …")
    build_var_figures(series_list, raw_results, models_dir, processed_dir, figures_dir)

    log.info("Building forecast-vs-observed figures …")
    build_forecast_vs_observed(series_list, cfg, models_dir, processed_dir, figures_dir)

    log.info("══ build_figures complete — outputs in %s ══", figures_dir)


def main():
    parser = argparse.ArgumentParser(description="Build paper figures")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
