"""tests/test_garch_init.py — Section 6: GARCH(1,1)-t -> LSTM init."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.garch_init import (
    logit,
    garch_initial_cell_state,
    garch_lstm_weight_arrays,
    garch_reference_path,
    verify_garch_path_reproduction,
    load_garch_params,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SERIES = ["BTC-USD", "ETH-USD", "DJIA", "SP500"]


def test_logit_inverts_sigmoid():
    for p in (0.01, 0.1, 0.5, 0.83, 0.99):
        assert abs(1.0 / (1.0 + np.exp(-logit(p))) - p) < 1e-9


def test_initial_cell_state_uses_formula_when_stationary():
    state = garch_initial_cell_state(omega=0.02, alpha=0.17, beta=0.82, sigma2_train=1.3)
    assert not state["used_fallback"]
    assert abs(state["c0"] - 0.02 / (1 - 0.17 - 0.82)) < 1e-10


def test_initial_cell_state_falls_back_at_igarch_boundary():
    state = garch_initial_cell_state(omega=0.16, alpha=0.11, beta=0.89, sigma2_train=13.77)
    assert state["used_fallback"]
    assert state["c0"] == pytest.approx(13.77)


def test_weight_arrays_encode_garch_params_on_unit_zero():
    alpha, beta, omega, sigma2_train = 0.15, 0.80, 0.05, 2.0
    units = 4
    w = garch_lstm_weight_arrays(alpha, beta, omega, sigma2_train, units=units,
                                  rng=np.random.default_rng(0))
    kernel, rk, bias = w["kernel"], w["recurrent_kernel"], w["bias"]
    assert kernel.shape == (1, 4 * units)
    assert rk.shape == (units, 4 * units)
    assert bias.shape == (4 * units,)

    i0, f0, c0b, o0 = 0, units, 2 * units, 3 * units
    assert bias[i0] == pytest.approx(logit(alpha))
    assert bias[f0] == pytest.approx(logit(beta))
    assert kernel[0, c0b] == pytest.approx(sigma2_train)
    assert bias[c0b] == pytest.approx(omega / alpha)
    assert bias[o0] == pytest.approx(5.0)
    # Unit 0's recurrent kernel row/cols stay exactly zero (no h_{t-1} feedback).
    assert rk[0, i0] == 0.0 and rk[0, f0] == 0.0 and rk[0, c0b] == 0.0 and rk[0, o0] == 0.0
    assert (rk[:, i0] == 0.0).all()


def test_weight_arrays_rejects_nonpositive_alpha():
    with pytest.raises(ValueError):
        garch_lstm_weight_arrays(alpha=0.0, beta=0.8, omega=0.1, sigma2_train=1.0, units=1)


def test_reference_path_matches_hand_computation():
    alpha, beta, omega = 0.1, 0.8, 0.05
    eps2 = np.array([1.0, 2.0, 0.5])
    c0 = 1.0
    path = garch_reference_path(alpha, beta, omega, eps2, c0)
    expected = []
    prev = c0
    for e in eps2:
        prev = omega + alpha * e + beta * prev
        expected.append(prev)
    assert np.allclose(path, expected)


def test_verify_garch_path_reproduction_synthetic_pass():
    """
    A self-consistent synthetic series (built from the exact recursion the
    probe implements) must PASS with error close to the deliberate o_t~=1
    budget (~0.67%), regardless of any real data.
    """
    rng = np.random.default_rng(0)
    alpha, beta, omega = 0.1, 0.8, 0.05
    sigma2_train = 1.0
    n = 300
    eps2 = rng.exponential(1.0, n)
    c0 = omega / (1 - alpha - beta)
    x_scaled = eps2 / sigma2_train

    res = verify_garch_path_reproduction(
        "SYNTH", alpha=alpha, beta=beta, omega=omega, sigma2_train=sigma2_train,
        x_scaled_train=x_scaled, eps2_raw_train=eps2,
    )
    assert res["verdict"] == "PASS"
    assert res["mean_relative_error"] < 0.01
    assert res["mean_relative_error"] == pytest.approx(1 - 1 / (1 + np.exp(-5.0)), abs=2e-3)


@pytest.mark.parametrize("series", _SERIES)
def test_garch_init_passes_for_all_project_series(series):
    """
    Regression guard: the actual four project series (using their real
    saved GARCH(1,1)-t estimates, scaler, and eps2 proxy) must PASS the
    Section 6 path-reproduction check. Skips if the required saved
    artifacts aren't present (e.g. a fresh checkout before `make models`).
    """
    models_dir = _REPO_ROOT / "outputs" / "models"
    processed_dir = _REPO_ROOT / "data" / "processed"
    params_path = models_dir / "GARCH11" / series / "params.json"
    scaler_path = processed_dir / series / "scaler.json"
    if not params_path.exists() or not scaler_path.exists():
        pytest.skip(f"GARCH11 params or scaler not found for {series}; run `make models` first.")

    params = load_garch_params(series, models_dir)
    train_eps2 = pd.read_csv(
        processed_dir / series / "train_eps2.csv", index_col=0
    ).iloc[:, 0].values.astype(float)
    scaler = json.loads(scaler_path.read_text())
    from src.data.scaling import transform as scaling_transform
    x_scaled_train = scaling_transform(train_eps2, scaler)

    res = verify_garch_path_reproduction(
        series,
        alpha=params["alpha"], beta=params["beta"], omega=params["omega"],
        sigma2_train=scaler["sigma2_train"],
        x_scaled_train=x_scaled_train,
        eps2_raw_train=train_eps2,
    )
    assert res["verdict"] == "PASS", res
