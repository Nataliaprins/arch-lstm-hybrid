"""
bootstrap.py — Bootstrap confidence intervals for MSE and MAE.

Method: block bootstrap (circular), B=2000 replications.
Reports 95% percentile-interval [P2.5, P97.5].
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def bootstrap_ci(
    sigma2_hat: np.ndarray,
    eps2:       np.ndarray,
    metric_fn,                       # callable(sigma2_hat, eps2) → float
    B:          int = 2000,
    block_size: int = 20,
    alpha:      float = 0.05,
    seed:       int = 42,
) -> tuple[float, float, float]:
    """
    Block-bootstrap confidence interval for a scalar metric.

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    rng  = np.random.default_rng(seed)
    T    = len(eps2)
    mask = np.isfinite(sigma2_hat) & np.isfinite(eps2)
    s2   = sigma2_hat[mask]
    e2   = eps2[mask]
    T    = len(e2)

    point = metric_fn(s2, e2)
    boots = np.empty(B)

    n_blocks = int(np.ceil(T / block_size))
    for b in range(B):
        starts  = rng.integers(0, T, size=n_blocks)
        idx     = np.concatenate(
            [np.arange(st, st + block_size) % T for st in starts]
        )[:T]
        boots[b] = metric_fn(s2[idx], e2[idx])

    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def bootstrap_ci_all(
    sigma2_hat: np.ndarray,
    eps2:       np.ndarray,
    B:          int = 2000,
    block_size: int = 20,
    alpha:      float = 0.05,
    seed:       int = 42,
) -> dict:
    """
    Return 95% bootstrap CI for both MSE and MAE.
    """
    from src.eval.metrics import mse as _mse, mae as _mae

    mse_pt, mse_lo, mse_hi = bootstrap_ci(
        sigma2_hat, eps2, _mse, B, block_size, alpha, seed
    )
    mae_pt, mae_lo, mae_hi = bootstrap_ci(
        sigma2_hat, eps2, _mae, B, block_size, alpha, seed + 1
    )
    return {
        "MSE":     mse_pt,
        "MSE_lo":  mse_lo,
        "MSE_hi":  mse_hi,
        "MAE":     mae_pt,
        "MAE_lo":  mae_lo,
        "MAE_hi":  mae_hi,
    }
