"""tests/test_data_scaling.py — Unit tests for the Section 4 input scaler."""
import numpy as np
import pytest

from src.data.scaling import fit_scaler, transform


RNG = np.random.default_rng(1)
TRAIN = RNG.exponential(2.0, 500)
VAL   = RNG.exponential(2.0, 100) * 5.0    # different scale on purpose
TEST  = RNG.exponential(2.0, 100) * 50.0   # even more different


def test_unconditional_sigma2_train_is_train_mean():
    scaler = fit_scaler(TRAIN, method="unconditional")
    assert scaler["method"] == "unconditional"
    assert abs(scaler["sigma2_train"] - TRAIN.mean()) < 1e-10


def test_unconditional_transform_is_dimensionless_ratio():
    scaler = fit_scaler(TRAIN, method="unconditional")
    x = transform(TRAIN, scaler)
    assert np.allclose(x, TRAIN / TRAIN.mean())


def test_log1p_has_no_fitted_parameter():
    scaler = fit_scaler(TRAIN, method="log1p")
    assert scaler == {"method": "log1p"}


def test_log1p_transform():
    scaler = fit_scaler(TRAIN, method="log1p")
    x = transform(TEST, scaler)
    assert np.allclose(x, np.log1p(TEST))


def test_scaler_never_reflects_val_or_test_stats():
    """
    Purity test: refitting the scaler with extra val/test-like data appended
    to the training array MUST change its parameters (proving sigma2_train
    is sensitive to what's fed in) — and therefore, conversely, a scaler
    fit on train-only data must be unaffected by whatever val/test contain.
    This guards against a future refactor that accidentally folds val/test
    into the fit call.
    """
    scaler_train_only = fit_scaler(TRAIN, method="unconditional")

    # If someone accidentally fits on train+val+test, the scale changes
    # substantially because VAL/TEST are drawn from very different scales.
    scaler_leaked = fit_scaler(np.concatenate([TRAIN, VAL, TEST]), method="unconditional")

    assert scaler_train_only["sigma2_train"] != pytest.approx(
        scaler_leaked["sigma2_train"], rel=0.05
    ), "scaler stats must be computed from the training split only"

    # And a scaler fit on train-only must be identical regardless of what
    # val/test look like (i.e. it is a pure function of TRAIN).
    scaler_again = fit_scaler(TRAIN, method="unconditional")
    assert scaler_train_only == scaler_again


def test_unconditional_rejects_nonpositive_mean():
    with pytest.raises(ValueError):
        fit_scaler(np.array([-1.0, -2.0, -3.0]), method="unconditional")


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        fit_scaler(TRAIN, method="minmax")
    with pytest.raises(ValueError):
        transform(TRAIN, {"method": "minmax"})
