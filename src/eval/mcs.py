"""
mcs.py — Model Confidence Set (Hansen, Lunde & Nason, 2011).

Level 90%, B=10 000 block-bootstrap replications (config: mcs.block_size=20).

Reference
---------
Hansen, P.R., Lunde, A., & Nason, J.M. (2011). "The Model Confidence Set."
Econometrica, 79(2), 453–497.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# MCS algorithm
# ──────────────────────────────────────────────────────────────────────────────

def _block_bootstrap_sample(
    losses: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Circular block bootstrap for loss matrix.

    losses : (T, M) — T observations, M models
    Returns a (T, M) bootstrapped sample.
    """
    T, M  = losses.shape
    n_blocks = int(np.ceil(T / block_size))
    starts   = rng.integers(0, T, size=n_blocks)
    indices  = np.concatenate(
        [np.arange(s, s + block_size) % T for s in starts]
    )[:T]
    return losses[indices]


def _t_max_stat(d_mean: np.ndarray, d_se: np.ndarray) -> float:
    """t_max elimination statistic."""
    with np.errstate(invalid="ignore"):
        t = d_mean / d_se
        t = np.where(np.isfinite(t), t, -np.inf)
    return float(np.max(t))


def _t_range_stat(d_mean: np.ndarray, d_se: np.ndarray) -> float:
    """t_range statistic (alternative)."""
    with np.errstate(invalid="ignore"):
        t = d_mean / d_se
        t = np.where(np.isfinite(t), t, 0.0)
    return float(np.max(t) - np.min(t))


def mcs(
    losses:     np.ndarray,
    model_names: list[str],
    alpha:      float = 0.10,
    B:          int   = 10_000,
    block_size: int   = 20,
    seed:       int   = 42,
    statistic:  str   = "Tmax",
) -> dict[str, bool]:
    """
    Compute the Model Confidence Set.

    Parameters
    ----------
    losses      : (T, M) — row = observation, col = model loss
    model_names : list of M model names
    alpha       : exclusion level (default 0.10 → 90% MCS)
    B           : bootstrap replications
    block_size  : stationary block length
    seed        : random seed for reproducibility
    statistic   : "Tmax" | "Trange"

    Returns
    -------
    in_mcs : {model_name: bool}  True if the model survives in the 90% MCS
    """
    assert losses.shape[1] == len(model_names), "losses columns must match model_names"
    rng = np.random.default_rng(seed)

    active      = list(range(len(model_names)))
    surviving   = {i: False for i in range(len(model_names))}
    loss_matrix = losses.copy()

    while len(active) > 1:
        M_a     = len(active)
        sub     = loss_matrix[:, active]  # (T, M_a)

        # Relative performance: d̄_{i·} = T⁻¹ Σ_t Σ_j (loss_it − loss_jt) / M_a
        d_mean, d_var = _relative_stats(sub)

        # Bootstrap distribution under H₀
        t_stat_obs = _t_max_stat(d_mean, d_var) if statistic == "Tmax" else _t_range_stat(d_mean, d_var)
        boot_stats = np.empty(B)
        for b in range(B):
            sub_b      = _block_bootstrap_sample(sub, block_size, rng)
            d_m_b, d_v_b = _relative_stats(sub_b)
            # Centre under H₀
            d_m_b = d_m_b - d_mean
            boot_stats[b] = (
                _t_max_stat(d_m_b, d_var) if statistic == "Tmax"
                else _t_range_stat(d_m_b, d_var)
            )

        p_val = float(np.mean(boot_stats >= t_stat_obs))

        if p_val >= alpha:
            # All active models survive → all are in MCS
            for i in active:
                surviving[i] = True
            break
        else:
            # Eliminate the model with the worst (highest) relative loss
            worst_idx = int(np.argmax(d_mean))
            active.pop(worst_idx)

    # If only one model remains
    if len(active) == 1:
        surviving[active[0]] = True

    return {model_names[i]: surviving[i] for i in range(len(model_names))}


def _relative_stats(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute d̄_i (mean relative loss) and its SE for each model in sub.
    sub : (T, M)
    """
    T, M = sub.shape
    # d̄_i = (1/M) Σ_j (T⁻¹ Σ_t (sub_it − sub_jt))  = sub_i_mean − overall_mean
    row_mean   = sub.mean(axis=0)            # (M,)
    d_mean     = row_mean - row_mean.mean()  # relative mean (M,)

    # Bootstrap SE not available here; use sample SE
    # SE(d̄_i) ≈ std(sub_i − sub_mean_all) / √T
    overall    = sub.mean(axis=1, keepdims=True)  # (T, 1)
    diff       = sub - overall                    # (T, M)
    d_std      = diff.std(axis=0, ddof=1)         # (M,)
    d_se       = np.maximum(d_std / np.sqrt(T), 1e-12)

    return d_mean, d_se
