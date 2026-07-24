"""
metrics.py — OOS evaluation metrics for volatility forecasts.

All metrics are evaluated against the proxy ε²_t (centred squared return).

Functions
---------
  mse, rmse, mae, r2_score   — standard point forecast metrics
  qlike                      — QLIKE loss (robust to proxy noise)
  ll_t_oos                   — Student-t log-likelihood OOS (column LL_t(OOS))
  delta_pct                  — percentage improvement vs. GARCH(1,1) benchmark
  compute_all                — convenience wrapper returning a dict
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Point forecast metrics
# ──────────────────────────────────────────────────────────────────────────────

def mse(sigma2_hat: np.ndarray, eps2: np.ndarray) -> float:
    """Mean squared error: E[(σ̂²_t − ε²_t)²]."""
    mask = _valid(sigma2_hat, eps2)
    return float(np.mean((sigma2_hat[mask] - eps2[mask]) ** 2))


def rmse(sigma2_hat: np.ndarray, eps2: np.ndarray) -> float:
    return float(np.sqrt(mse(sigma2_hat, eps2)))


def mae(sigma2_hat: np.ndarray, eps2: np.ndarray) -> float:
    """Mean absolute error: E[|σ̂²_t − ε²_t|]."""
    mask = _valid(sigma2_hat, eps2)
    return float(np.mean(np.abs(sigma2_hat[mask] - eps2[mask])))


def r2_score(sigma2_hat: np.ndarray, eps2: np.ndarray) -> float:
    """R² of the OOS forecast vs. the proxy."""
    mask = _valid(sigma2_hat, eps2)
    ss_res = np.sum((eps2[mask] - sigma2_hat[mask]) ** 2)
    ss_tot = np.sum((eps2[mask] - eps2[mask].mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ──────────────────────────────────────────────────────────────────────────────
# QLIKE
# ──────────────────────────────────────────────────────────────────────────────

def qlike(sigma2_hat: np.ndarray, eps2: np.ndarray) -> float:
    """
    QLIKE = T⁻¹ Σ_t [ ln σ̂²_t + ε²_t / σ̂²_t ]

    Robust to noise in the proxy ε²_t; used as the loss for DM tests.
    """
    mask = _valid(sigma2_hat, eps2)
    s2   = np.maximum(sigma2_hat[mask], 1e-10)
    return float(np.mean(np.log(s2) + eps2[mask] / s2))


# ──────────────────────────────────────────────────────────────────────────────
# Student-t OOS log-likelihood (column LL_t(OOS) in Tables 4–7)
# ──────────────────────────────────────────────────────────────────────────────

def ll_t_oos(
    sigma2_hat: np.ndarray,
    eps2:       np.ndarray,
    nu:         float = 5.0,
) -> float:
    """
    Average per-observation Student-t log-likelihood evaluated OOS.
    Uses the standardised Student-t density with variance σ²_t for any ν > 2.

        ll_t = log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(π(ν−2))
               − ½ log σ²_t − ½(ν+1) log(1 + ε²_t / (σ²_t (ν−2)))
    """
    from src.losses.hybrid_student_t import student_t_ll_mean
    mask = _valid(sigma2_hat, eps2)
    return student_t_ll_mean(eps2[mask], sigma2_hat[mask], nu=nu)


# ──────────────────────────────────────────────────────────────────────────────
# Delta %
# ──────────────────────────────────────────────────────────────────────────────

def delta_pct(metric_model: float, metric_benchmark: float) -> float:
    """
    Percentage change relative to the benchmark (GARCH(1,1)).
    Negative = improvement.

        Δ% = 100 × (metric_model − metric_benchmark) / |metric_benchmark|
    """
    if not np.isfinite(metric_benchmark) or metric_benchmark == 0:
        return float("nan")
    return float(100.0 * (metric_model - metric_benchmark) / abs(metric_benchmark))


# ──────────────────────────────────────────────────────────────────────────────
# QLIKE loss array (per-observation, for DM test)
# ──────────────────────────────────────────────────────────────────────────────

def qlike_array(sigma2_hat: np.ndarray, eps2: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE loss (for Diebold–Mariano computation)."""
    s2 = np.maximum(sigma2_hat, 1e-10)
    return np.log(s2) + eps2 / s2


def mse_array(sigma2_hat: np.ndarray, eps2: np.ndarray) -> np.ndarray:
    """Per-observation squared error."""
    return (sigma2_hat - eps2) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Batch wrapper
# ──────────────────────────────────────────────────────────────────────────────

def compute_all(
    sigma2_hat:          np.ndarray,
    eps2:                np.ndarray,
    sigma2_hat_benchmark: np.ndarray | None = None,
    nu:                  float = 5.0,
) -> dict:
    """
    Compute all OOS metrics at once.

    Returns
    -------
    dict with keys: MSE, RMSE, MAE, R2, QLIKE, LL_t_OOS, Delta_MSE, Delta_MAE
    """
    m = {
        "MSE":      mse(sigma2_hat, eps2),
        "RMSE":     rmse(sigma2_hat, eps2),
        "MAE":      mae(sigma2_hat, eps2),
        "R2":       r2_score(sigma2_hat, eps2),
        "QLIKE":    qlike(sigma2_hat, eps2),
        "LL_t_OOS": ll_t_oos(sigma2_hat, eps2, nu=nu),
    }
    if sigma2_hat_benchmark is not None:
        m["Delta_MSE"] = delta_pct(m["MSE"], mse(sigma2_hat_benchmark, eps2))
        m["Delta_MAE"] = delta_pct(m["MAE"], mae(sigma2_hat_benchmark, eps2))
    else:
        m["Delta_MSE"] = float("nan")
        m["Delta_MAE"] = float("nan")
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

def _valid(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask: True where all arrays have finite, positive values."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask
