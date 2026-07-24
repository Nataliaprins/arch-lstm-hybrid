"""
base.py — Shared interface for ALL volatility models (econometric and neural).

Every model implements:
    fit(train_eps)                      → trains / estimates on training residuals
    predict(test_eps, train_eps)        → returns σ²_t OOS forecasts (n_test,)
    get_params()                        → dict of estimated parameters
    get_fit_info()                      → LL in-sample, AIC, BIC, convergence …
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVolatilityModel(ABC):
    """Abstract base: fit / predict common interface."""

    name:   str = "BaseModel"
    family: str = "Base"

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, train_eps: np.ndarray) -> None:
        """Fit / train on training residuals ε_t  (scale: 100 × log-ret)."""

    @abstractmethod
    def predict(self, test_eps: np.ndarray, train_eps: np.ndarray) -> np.ndarray:
        """
        1-step-ahead OOS σ²_t forecasts for the test period.

        Parameters
        ----------
        test_eps  : ε_t for the test period,  shape (n_test,)
        train_eps : ε_t for the train period, shape (n_train,)
                    Needed by recursive models to bridge the split boundary.

        Returns
        -------
        sigma2_hat : (n_test,) — conditional variance σ²_t (same scale as ε²_t)
        """

    # ------------------------------------------------------------------
    # Optional (override in subclasses)
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return dict of estimated / selected parameters."""
        return {}

    def get_fit_info(self) -> dict:
        """Return in-sample LL, AIC, BIC, convergence flags, timing, etc."""
        return {}
