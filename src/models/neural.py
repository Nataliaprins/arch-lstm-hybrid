"""
neural.py — Neural-network volatility model architecture builders.

Each public function builds and returns a compiled Keras model that accepts
inputs of shape  (batch, W, n_features)  and outputs a scalar σ²_t.

The build functions take a hyperparameter dict (from the tuner or best_hparams)
so that every architecture is driven by the same search space.

Models implemented
------------------
  build_lstm_sse          — plain LSTM, MSE / SSE loss
  build_lstm_t_student    — plain LSTM, hybrid Student-t loss (proposed)
  build_cnn_lstm          — Conv1D + LSTM
  build_lstm_attention    — LSTM + additive attention (Bahdanau)
  build_tcn               — Temporal Convolutional Network (dilated causal)
  build_transformer       — encoder-only transformer
  build_nn_garch          — dense net augmenting the GARCH recursion
  SVRGARCHModel           — scikit-learn SVR (sklearn, not Keras)

All Keras builders share the same output reparametrization (Section 3):
the output layer has NO activation and emits u_t = log σ̂²_t directly.
σ̂²_t is never computed inside the model graph; it is derived only where
needed — in the training loss (src.losses.hybrid_student_t.make_hybrid_loss
/ make_variance_mse_loss) and, at inference time, via
src.losses.hybrid_student_t.sigma2_from_log_var(model.predict(...)) —
as exp(clip(u_t, -20, 20)). The previous softplus-on-variance activation
was poorly conditioned when the target spans multiple orders of magnitude
(the case for crypto), which this reparametrization avoids.
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Seed / initialiser helper
# ──────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    """Fix TF, NumPy, Python seeds for a single training run."""
    import os, random, tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ──────────────────────────────────────────────────────────────────────────────
# build_lstm_sse  (LSTM-SSE — MSE baseline)
# ──────────────────────────────────────────────────────────────────────────────

def build_lstm_sse(hp: dict) -> "tf.keras.Model":
    """
    Single-layer LSTM followed by a Dense output layer.
    Loss: MSE.

    hp keys used: lstm_units, dropout, learning_rate (optional, default 1e-3)
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_variance_mse_loss

    units = hp["lstm_units"]
    drop  = hp.get("dropout", 0.0)
    lr    = hp.get("learning_rate", 1e-3)
    W     = hp["window_size"]

    inp   = tf.keras.Input(shape=(W, 1), name="eps_window")
    x     = tf.keras.layers.LSTM(units, dropout=drop, recurrent_dropout=0.0,
                                  kernel_initializer="glorot_uniform")(inp)
    x     = tf.keras.layers.Dropout(drop)(x)
    out   = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    model = tf.keras.Model(inp, out, name="LSTM-SSE")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_variance_mse_loss())
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_lstm_t_student  (LSTM-SSE-t-Student — proposed model)
# ──────────────────────────────────────────────────────────────────────────────

def _make_learned_nu_model(base_model, rho_init: float, lam: float,
                            s_sse: float = 1.0, s_t: float = 1.0, name: str = "LSTM-SSE-t-Student"):
    """
    Section 8: wraps a base model (predicting u_t = log σ̂²_t) with a
    trainable, global Student-t ν = 2 + softplus(ρ), via a custom
    train_step/test_step so ν participates in gradient descent alongside
    the network's weights. predict()/call() are unchanged (same
    (batch, 1) log_sigma2 output as the fixed-ν path) — only the loss
    computation and optimizer step differ, which keeps every downstream
    model.predict(...) call site (sigma2_from_log_var conversion, etc.)
    working exactly as it does for the fixed-ν path.

    A local class (not module-level) matching this file's lazy-TF-import
    convention: tf.keras.Model must already be resolvable when the class
    body evaluates.
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import hybrid_loss_components_tf

    class _LearnedNuModel(tf.keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.base_model = base_model
            self.rho = self.add_weight(
                name="nu_rho", shape=(), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(float(rho_init)), trainable=True,
            )
            self.lam_val = float(np.clip(lam, 0.0, 1.0))
            self.s_sse_val = float(s_sse)
            self.s_t_val = float(s_t)

        def call(self, inputs, training=False):
            return self.base_model(inputs, training=training)

        def nu(self):
            return 2.0 + tf.nn.softplus(self.rho)

        def _compute_loss(self, y_true, y_pred):
            L_sse, L_t = hybrid_loss_components_tf(y_true, y_pred, self.nu(), self.s_sse_val, self.s_t_val)
            return (1.0 - self.lam_val) * L_sse + self.lam_val * L_t

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self._compute_loss(y, y_pred)
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
            return {"loss": loss}

        def test_step(self, data):
            x, y = data
            y_pred = self(x, training=False)
            loss = self._compute_loss(y, y_pred)
            return {"loss": loss}

    return _LearnedNuModel(name=name)


def build_lstm_t_student(hp: dict) -> "tf.keras.Model":
    """
    Same LSTM architecture as LSTM-SSE but trained with the hybrid
    (1−λ)·MSE/s_sse + λ·NLL_Student-t/s_t loss.

    Extra hp keys: nu (degrees of freedom, used when hp["nu_mode"] is
    not "learned"), lam (λ mixing weight), s_sse / s_t (frozen
    normalization scales — see src.losses.hybrid_student_t.
    compute_loss_scales; default 1.0/1.0 reproduces the un-normalized
    loss).

    hp["nu_mode"] (Section 8):
      "fixed" (default) — ν = hp["nu"], a constant (either sampled by
               the hyperparameter search, in which case it was
               previously selected by validation forecast loss -- see
               "likelihood_search" below for the corrected alternative
               -- or set directly by the caller).
      "learned" — ν = 2 + softplus(ρ) is a trainable model parameter,
               ρ initialized from hp["nu_rho_init"] (the value implied
               by this series' GARCH-t ν̂). Returns a LearnedNuModel
               wrapping the usual functional model; predict() output is
               unchanged.
      Any other value of hp["nu"] set by an outer "likelihood_search"
      selection (Section 8) is just a fixed ν chosen by OOS likelihood
      rather than validation MSE/hybrid-loss -- from this function's
      point of view that is identical to "fixed" (nu_mode stays "fixed"
      or is omitted; the search happens one level up, in
      src.tuning.tune_and_train).

    hp["init"] selects the LSTM weight initialization (Section 6):
      "glorot" (default) — standard Keras init, tanh activation.
      "garch"  — single-unit LSTM initialized from the GARCH(1,1)-t MLE
                 solution of this series (src.models.garch_init), with a
                 linear LSTM activation and a fixed (non-trainable)
                 initial cell state c0. Requires hp["garch_alpha"],
                 hp["garch_beta"], hp["garch_omega"],
                 hp["garch_sigma2_train"] (only valid when
                 data.input_scaling="unconditional" — see
                 src.models.garch_init module docstring for the mapping).
                 hp["lstm_units"] is ignored (forced to 1): a linear
                 activation has no saturation to bound extra hidden
                 units, so random-initialized capacity beyond the single
                 GARCH-tracking unit is numerically unstable over the
                 window length (verified empirically -- it diverges
                 within one epoch). Proposition 2's construction is a
                 single scalar recursion; that is what this reproduces.
    """
    import numpy as np
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_hybrid_loss

    drop     = hp.get("dropout", 0.0)
    lr       = hp.get("learning_rate", 1e-3)
    lam      = hp["lam"]
    W        = hp["window_size"]
    nu_mode  = hp.get("nu_mode", "fixed")
    s_sse = hp.get("s_sse", 1.0)
    s_t   = hp.get("s_t", 1.0)
    init  = hp.get("init", "glorot")
    units = 1 if init == "garch" else hp["lstm_units"]

    inp = tf.keras.Input(shape=(W, 1), name="eps_window")

    if init == "garch":
        from src.models.garch_init import (
            garch_lstm_weight_arrays, garch_initial_cell_state,
            make_constant_initial_state_layer,
        )
        alpha = hp["garch_alpha"]
        beta = hp["garch_beta"]
        omega = hp["garch_omega"]
        sigma2_train = hp["garch_sigma2_train"]
        c0 = garch_initial_cell_state(omega, alpha, beta, sigma2_train)["c0"]

        h0_vec = np.zeros(units, dtype=np.float32)
        c0_vec = np.full(units, c0, dtype=np.float32)
        h0 = make_constant_initial_state_layer(h0_vec, name="h0_init")(inp)
        c0_state = make_constant_initial_state_layer(c0_vec, name="c0_init")(inp)

        lstm_layer = tf.keras.layers.LSTM(
            units, activation="linear", recurrent_activation="sigmoid",
            dropout=drop, recurrent_dropout=0.0,
        )
        x = lstm_layer(inp, initial_state=[h0, c0_state])
        w = garch_lstm_weight_arrays(alpha, beta, omega, sigma2_train, units=units)
        lstm_layer.set_weights([w["kernel"], w["recurrent_kernel"], w["bias"]])
    else:
        x = tf.keras.layers.LSTM(units, dropout=drop, recurrent_dropout=0.0,
                                  kernel_initializer="glorot_uniform")(inp)

    x     = tf.keras.layers.Dropout(drop)(x)
    out   = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    base_model = tf.keras.Model(inp, out, name="LSTM-SSE-t-Student")

    if nu_mode == "learned":
        model = _make_learned_nu_model(
            base_model, rho_init=hp["nu_rho_init"], lam=lam, s_sse=s_sse, s_t=s_t,
            name="LSTM-SSE-t-Student",
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    else:
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=make_hybrid_loss(nu=hp["nu"], lam=lam, s_sse=s_sse, s_t=s_t),
        )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_cnn_lstm  (CNN-LSTM)
# ──────────────────────────────────────────────────────────────────────────────

def build_cnn_lstm(hp: dict) -> "tf.keras.Model":
    """
    Conv1D feature extraction → LSTM → Dense.
    hp extras: n_filters (default = lstm_units // 2), kernel_size (default 3)
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_variance_mse_loss

    units       = hp["lstm_units"]
    drop        = hp.get("dropout", 0.0)
    lr          = hp.get("learning_rate", 1e-3)
    n_filters   = hp.get("n_filters", max(8, units // 2))
    kernel_size = hp.get("kernel_size", 3)
    W           = hp["window_size"]

    inp = tf.keras.Input(shape=(W, 1))
    x   = tf.keras.layers.Conv1D(n_filters, kernel_size, padding="causal",
                                  activation="relu",
                                  kernel_initializer="glorot_uniform")(inp)
    x   = tf.keras.layers.Conv1D(n_filters, kernel_size, padding="causal",
                                  activation="relu",
                                  kernel_initializer="glorot_uniform")(x)
    x   = tf.keras.layers.LSTM(units, dropout=drop,
                                 kernel_initializer="glorot_uniform")(x)
    x   = tf.keras.layers.Dropout(drop)(x)
    out = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    model = tf.keras.Model(inp, out, name="CNN-LSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_variance_mse_loss())
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_lstm_attention  (LSTM + additive attention)
# ──────────────────────────────────────────────────────────────────────────────

def build_lstm_attention(hp: dict) -> "tf.keras.Model":
    """LSTM with return_sequences=True + additive (Bahdanau-style) attention."""
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_variance_mse_loss

    units = hp["lstm_units"]
    drop  = hp.get("dropout", 0.0)
    lr    = hp.get("learning_rate", 1e-3)
    W     = hp["window_size"]

    inp  = tf.keras.Input(shape=(W, 1))
    lstm_out = tf.keras.layers.LSTM(
        units, return_sequences=True, dropout=drop,
        kernel_initializer="glorot_uniform"
    )(inp)

    # Bahdanau-style additive attention
    score  = tf.keras.layers.Dense(1, activation="tanh")(lstm_out)   # (B, W, 1)
    score  = tf.keras.layers.Flatten()(score)                         # (B, W)
    attn   = tf.keras.layers.Activation("softmax")(score)             # (B, W)
    attn   = tf.keras.layers.Reshape((W, 1))(attn)                    # (B, W, 1)
    ctx    = tf.keras.layers.Multiply()([lstm_out, attn])              # (B, W, units)
    ctx    = tf.keras.layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1)
    )(ctx)                                                              # (B, units)

    x   = tf.keras.layers.Dropout(drop)(ctx)
    out = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    model = tf.keras.Model(inp, out, name="LSTM-Attention")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_variance_mse_loss())
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_tcn  (Temporal Convolutional Network)
# ──────────────────────────────────────────────────────────────────────────────

def _tcn_residual_block(x, n_filters, kernel_size, dilation, drop):
    """Single TCN residual block with causal dilated convolution."""
    import tensorflow as tf

    res = x
    for _ in range(2):
        x = tf.keras.layers.Conv1D(
            n_filters, kernel_size, padding="causal",
            dilation_rate=dilation, activation=None,
            kernel_initializer="glorot_uniform",
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SpatialDropout1D(drop)(x)

    # Residual connection (match channels if needed)
    if res.shape[-1] != n_filters:
        res = tf.keras.layers.Conv1D(n_filters, 1, padding="same")(res)
    x = tf.keras.layers.Add()([x, res])
    return x


def build_tcn(hp: dict) -> "tf.keras.Model":
    """
    Temporal Convolutional Network with dilated causal convolutions.
    Receptive field = kernel_size × Σ dilations  ≥ window_size.
    Dilations: [1, 2, 4, 8] → RF = 3 × 15 = 45 > 22 for kernel_size=3.
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_variance_mse_loss

    units       = hp["lstm_units"]     # reused as n_filters
    drop        = hp.get("dropout", 0.0)
    lr          = hp.get("learning_rate", 1e-3)
    kernel_size = hp.get("kernel_size", 3)
    W           = hp["window_size"]
    dilations   = [1, 2, 4, 8]

    inp = tf.keras.Input(shape=(W, 1))
    x   = inp
    for d in dilations:
        x = _tcn_residual_block(x, units, kernel_size, d, drop)

    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    model = tf.keras.Model(inp, out, name="TCN")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_variance_mse_loss())
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_transformer  (encoder-only Transformer)
# ──────────────────────────────────────────────────────────────────────────────

def _positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding, shape (1, seq_len, d_model)."""
    import tensorflow as tf

    positions = np.arange(seq_len)[:, np.newaxis]         # (W, 1)
    dims      = np.arange(d_model)[np.newaxis, :]          # (1, d_model)
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], tf.float32)   # (1, W, d_model)


def build_transformer(hp: dict) -> "tf.keras.Model":
    """
    Encoder-only Transformer for 1-step-ahead volatility forecast.
    hp extras: n_heads (default 2), d_model (default lstm_units), ffn_dim (default 2×d_model)
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_variance_mse_loss

    d_model  = hp.get("d_model", hp["lstm_units"])
    n_heads  = hp.get("n_heads", 2)
    ffn_dim  = hp.get("ffn_dim", d_model * 2)
    drop     = hp.get("dropout", 0.0)
    lr       = hp.get("learning_rate", 1e-3)
    W        = hp["window_size"]

    inp   = tf.keras.Input(shape=(W, 1))
    # Project to d_model
    x     = tf.keras.layers.Dense(d_model, kernel_initializer="glorot_uniform")(inp)

    # Positional encoding (constant, not trainable)
    pe    = _positional_encoding(W, d_model)
    x     = x + pe

    # Multi-head self-attention block
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        dropout=drop,
    )(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn_out)

    # Feed-forward block
    ffn_out = tf.keras.layers.Dense(ffn_dim, activation="relu",
                                     kernel_initializer="glorot_uniform")(x)
    ffn_out = tf.keras.layers.Dense(d_model,
                                     kernel_initializer="glorot_uniform")(ffn_out)
    x       = tf.keras.layers.LayerNormalization()(x + ffn_out)
    x       = tf.keras.layers.Dropout(drop)(x)

    # Aggregate over time → scalar output
    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    model = tf.keras.Model(inp, out, name="Transformer")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_variance_mse_loss())
    return model


# ──────────────────────────────────────────────────────────────────────────────
# build_nn_garch  (Dense net augmenting the GARCH recursion — req. by Reviewer 2)
# ──────────────────────────────────────────────────────────────────────────────

def build_nn_garch(hp: dict) -> "tf.keras.Model":
    """
    Feed-forward network with two inputs:  (ε²_{t-1}, σ²_{t-1}_GARCH).
    The GARCH-filtered variance serves as a structured prior; the network
    learns residual corrections.

    Input shape: (batch, W, 2)  — channel 0: ε_t, channel 1: σ²_t_garch
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_variance_mse_loss

    units = hp["lstm_units"]   # repurposed as hidden units
    drop  = hp.get("dropout", 0.0)
    lr    = hp.get("learning_rate", 1e-3)
    W     = hp["window_size"]

    inp = tf.keras.Input(shape=(W, 2))
    x   = tf.keras.layers.Flatten()(inp)
    for _ in range(2):
        x = tf.keras.layers.Dense(units, activation="relu",
                                   kernel_initializer="glorot_uniform")(x)
        x = tf.keras.layers.Dropout(drop)(x)
    out = tf.keras.layers.Dense(1, activation=None, name="log_sigma2")(x)

    model = tf.keras.Model(inp, out, name="NN-GARCH")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=make_variance_mse_loss())
    return model


# ──────────────────────────────────────────────────────────────────────────────
# SVRGARCHModel  (scikit-learn, not Keras)
# ──────────────────────────────────────────────────────────────────────────────

class SVRGARCHModel:
    """
    SVR trained on a window of ε²_t values.

    Scikit-learn SVR with RBF kernel.  Deterministic (no seed needed).
    The 'multi-seed' protocol reports mean=result, std=0 for this model.

    Parameters
    ----------
    C       : regularisation
    epsilon : SVR tube width
    gamma   : RBF kernel width ('scale' or float)
    window  : lookback window length
    """

    name   = "SVR-GARCH"
    family = "ML"

    def __init__(self, hp: dict):
        self.window = hp["window_size"]
        self.C      = hp.get("C", 1.0)
        self.eps    = hp.get("svr_epsilon", 0.1)
        self.gamma  = hp.get("gamma", "scale")
        self._svr   = None

    def _make_windows(self, eps2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = self.window
        X, y = [], []
        for t in range(W, len(eps2)):
            X.append(eps2[t - W: t])
            y.append(eps2[t])
        return np.array(X), np.array(y)

    def fit(self, train_eps2: np.ndarray) -> None:
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler

        X, y = self._make_windows(train_eps2)
        self._scaler_X = StandardScaler().fit(X)
        self._scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        X_sc = self._scaler_X.transform(X)
        y_sc = self._scaler_y.transform(y.reshape(-1, 1)).ravel()

        self._svr = SVR(kernel="rbf", C=self.C, epsilon=self.eps, gamma=self.gamma)
        self._svr.fit(X_sc, y_sc)

    def predict(self, train_eps2: np.ndarray, test_eps2: np.ndarray) -> np.ndarray:
        """Return σ²_hat for the test window."""
        full_eps2 = np.concatenate([train_eps2, test_eps2])
        n_train   = len(train_eps2)
        n_test    = len(test_eps2)
        W         = self.window

        preds = []
        for i in range(n_test):
            t = n_train + i
            x = full_eps2[t - W: t].reshape(1, -1)
            x_sc = self._scaler_X.transform(x)
            y_sc = self._svr.predict(x_sc)
            y_orig = self._scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()[0]
            preds.append(max(y_orig, 1e-8))

        return np.array(preds)
