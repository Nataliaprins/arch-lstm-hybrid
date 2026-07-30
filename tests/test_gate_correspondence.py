"""tests/test_gate_correspondence.py — Section 9.3: LSTM gate <-> GARCH correspondence."""
import numpy as np
import pytest

from src.eval.gate_correspondence import (
    lstm_forward_gates, regression_lstm_vs_garch, _sigmoid,
    compute_gate_statistics_one_seed,
)


def test_sigmoid_basic():
    assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert _sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert _sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_lstm_forward_gates_matches_keras():
    """
    The hand-rolled NumPy forward pass must reproduce tf.keras.layers.LSTM
    exactly (same weights, same inputs) -- this is the ground-truth
    parity check for the whole gate-extraction approach.
    """
    import tensorflow as tf

    units, W, n = 4, 6, 10
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, W, 1)).astype(np.float32)

    layer = tf.keras.layers.LSTM(units, activation="tanh", recurrent_activation="sigmoid",
                                  return_sequences=False)
    inp = tf.keras.Input(shape=(W, 1))
    out = layer(inp)
    model = tf.keras.Model(inp, out)
    h_keras = model.predict(X, verbose=0)

    kernel, recurrent_kernel, bias = layer.get_weights()
    gates = lstm_forward_gates(X, kernel, recurrent_kernel, bias, units, activation="tanh")

    assert np.allclose(gates["h"], h_keras, atol=1e-5)


def test_lstm_forward_gates_linear_activation():
    """Linear-activation variant (Section 6 GARCH-init cells) must also match Keras."""
    import tensorflow as tf

    units, W, n = 1, 5, 8
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, W, 1)).astype(np.float32)

    layer = tf.keras.layers.LSTM(units, activation="linear", recurrent_activation="sigmoid",
                                  return_sequences=False)
    inp = tf.keras.Input(shape=(W, 1))
    out = layer(inp)
    model = tf.keras.Model(inp, out)
    h_keras = model.predict(X, verbose=0)

    kernel, recurrent_kernel, bias = layer.get_weights()
    gates = lstm_forward_gates(X, kernel, recurrent_kernel, bias, units, activation="linear")

    assert np.allclose(gates["h"], h_keras, atol=1e-5)


def test_regression_lstm_vs_garch_near_identity_gives_b_near_one():
    """
    b=1 exactly (zero residual variance) is a numerically degenerate edge
    case for a HAC standard error. Add realistic noise instead, matching
    how this regression is actually used (two different models' forecasts,
    never identical), and check b is close to 1 with a sane, non-degenerate
    CI -- not that the CI happens to straddle 1.0 for one arbitrary noise
    draw, which is a ~95%-of-the-time property, not an always-true one.
    """
    rng = np.random.default_rng(2)
    sigma2_garch = rng.exponential(2.0, 500) + 1.0
    sigma2_lstm = sigma2_garch + rng.normal(0, 0.3, len(sigma2_garch))

    res = regression_lstm_vs_garch(sigma2_lstm, sigma2_garch)
    assert res["b"] == pytest.approx(1.0, abs=0.1)
    assert res["pearson_r"] > 0.95
    assert res["b_ci_low"] < res["b"] < res["b_ci_high"]
    assert (res["b_ci_high"] - res["b_ci_low"]) < 0.5


def test_regression_lstm_vs_garch_unrelated_series():
    rng = np.random.default_rng(3)
    sigma2_garch = rng.exponential(2.0, 500) + 1.0
    sigma2_lstm = rng.exponential(2.0, 500) + 1.0  # independent draw

    res = regression_lstm_vs_garch(sigma2_lstm, sigma2_garch)
    assert abs(res["pearson_r"]) < 0.3


def test_compute_gate_statistics_one_seed_learned_nu(tmp_path):
    """
    Reproduces the real run-time failure: with nu_mode="learned",
    build_lstm_t_student returns a _LearnedNuModel (a tf.keras.Model
    subclass), which Keras only considers "built" after a first forward
    call. Loading weights into an unbuilt subclassed model raises "You
    are loading weights into a model that has not yet been built." This
    test builds a tiny model, saves weights the same way
    _multiseed_train_and_predict does (model.save_weights after at least
    one call), then verifies gate extraction on a *freshly constructed,
    never-called* model succeeds.
    """
    import tensorflow as tf
    from src.models.neural import build_lstm_t_student

    hp = {
        "lstm_units": 1, "dropout": 0.0, "batch_size": 8,
        "learning_rate": 1e-3, "window_size": 5,
        "nu_mode": "learned", "nu_rho_init": 1.0, "lam": 0.5,
        "s_sse": 1.0, "s_t": 1.0,
    }
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, hp["window_size"], 1)).astype("float32")
    y = np.abs(rng.normal(size=16)).astype("float32")

    trained = build_lstm_t_student(hp)
    trained.fit(X, y, epochs=1, batch_size=hp["batch_size"], verbose=0)
    weights_path = tmp_path / "weights.weights.h5"
    trained.save_weights(str(weights_path))

    eps2_test = np.abs(rng.normal(size=16)).astype("float32")
    stats = compute_gate_statistics_one_seed(hp, weights_path, X, eps2_test)
    assert np.isfinite(stats["E_i"])
    assert np.isfinite(stats["E_f"])
