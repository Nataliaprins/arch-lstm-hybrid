"""
scaling.py — Input scaling for neural volatility models (Section 4
respecification).

Replaces the previous unscaled-raw-return input (and any MinMax-style
scaling) with a dimensionless, train-only-fit feature built from the
ε²_t proxy itself, so that:

  * every neural architecture's recurrent input lines up with the
    quantity the GARCH(1,1) recursion actually operates on
    (σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}) — required for the gate/
    parameter correspondence in Proposition 2 (Sections 6/7/9.3) to be
    structurally possible at all;
  * the input magnitude is comparable across markets (crypto vs. equity
    indices), instead of depending on each series' raw units.

Two methods, selected via config `data.input_scaling`:

  unconditional : x_t = ε²_t / σ̄²_train   (σ̄²_train = mean(ε²_train))
  log1p         : x_t = log1p(ε²_t)        (no fitted parameter)

The scaler is fit **once**, on the training split only, and persisted
(see src.data.build_dataset). It must never be refit or its stats
recomputed using validation/test data.
"""
from __future__ import annotations

import numpy as np

VALID_METHODS = ("unconditional", "log1p")


def fit_scaler(eps2_train: np.ndarray, method: str = "unconditional") -> dict:
    """
    Fit scaler statistics using ONLY the training-split ε²_t proxy.

    Returns a JSON-serializable dict; for "log1p" there is no fitted
    parameter (the transform is parameter-free), but the method is still
    recorded so `transform` and downstream code know which rule to apply.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown data.input_scaling method: {method!r} (expected one of {VALID_METHODS})")

    eps2_train = np.asarray(eps2_train, dtype=float)

    if method == "unconditional":
        sigma2_train = float(np.mean(eps2_train))
        if not np.isfinite(sigma2_train) or sigma2_train <= 0:
            raise ValueError(f"sigma2_train must be finite and positive, got {sigma2_train}")
        return {"method": "unconditional", "sigma2_train": sigma2_train}

    return {"method": "log1p"}


def transform(eps2: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply an already-fitted scaler to any split (train, val, or test)."""
    eps2 = np.asarray(eps2, dtype=float)
    method = scaler["method"]
    if method == "unconditional":
        return eps2 / scaler["sigma2_train"]
    if method == "log1p":
        return np.log1p(eps2)
    raise ValueError(f"Unknown data.input_scaling method: {method!r}")
