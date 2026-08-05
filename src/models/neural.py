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

Every builder EXCEPT build_lstm_t_student shares the same output
reparametrization (Section 3): the output layer has NO activation and
emits u_t = log σ̂²_t directly. σ̂²_t is never computed inside the model
graph; it is derived only where needed — in the training loss
(src.losses.hybrid_student_t.make_variance_mse_loss) and, at inference
time, via src.losses.hybrid_student_t.sigma2_from_log_var(model.predict(...))
— as exp(clip(u_t, -20, 20)). The previous softplus-on-variance
activation was poorly conditioned when the target spans multiple orders
of magnitude (the case for crypto), which this reparametrization avoids.

build_lstm_t_student (the proposed model) instead uses
activation="exponential" directly on its output layer, so σ̂²_t > 0 is
guaranteed by the architecture rather than by a downstream transform;
its raw predict() output IS σ̂²_t (see
src.losses.hybrid_student_t.sigma2_from_direct_output /
hybrid_loss_components_sigma2_tf / make_hybrid_loss_sigma2 /
make_hybrid_loss_variable_nu_sigma2). It also uses no GARCH-derived
inputs or initialization of any kind — see its own docstring.
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

def _make_adaptive_lr_schedule(hp: dict):
    """
    Warm-up (linear) + plain cosine decay learning-rate schedule, opt-in
    via hp["adaptive_lr"]=True.

    Originally (2026-07-31) this used cosine-decay-*with-restarts* (SGDR,
    Loshchilov & Hutter 2017), on the theory that periodically re-raising
    the LR could substitute for the mini-batch sampling noise that
    full_batch_training removes, helping escape a degenerate
    near-constant sigma2_t local optimum. Two things changed that:
    (1) validation runs found no consistent collapse benefit from the
    restarts; (2) each LR spike measurably worsens val_loss right after
    the restart, which -- combined with EarlyStopping's patience counting
    consecutive non-improving epochs -- kept resetting that counter and
    let many random-search trials run close to the full epoch budget
    instead of stopping early, making the restart schedule the likely
    cause of some series' (e.g. GOLD's) search phase taking far longer
    than others' with no offsetting benefit. A single monotonic decay to
    a floor lets val_loss settle predictably so patience can actually
    trigger. The short linear warm-up is kept, avoiding a large,
    potentially destabilizing first step straight from a random init.
    Step count is in optimizer iterations, which under full-batch
    training equals epochs.

    hp keys (all optional, sane defaults for a ~3000-epoch full-batch run):
      learning_rate  : peak LR after warm-up (default 1e-3)
      lr_warmup_steps : linear warm-up length in steps/epochs (default 20)
      lr_decay_steps   : length of the cosine decay to the floor (default 1000)
      lr_alpha           : floor LR as a fraction of peak_lr, held constant
                           for any remaining steps past lr_decay_steps (default 0.05)
    """
    import tensorflow as tf

    peak_lr      = float(hp.get("learning_rate", 1e-3))
    warmup_steps = float(max(int(hp.get("lr_warmup_steps", 20)), 1))
    decay_steps  = int(hp.get("lr_decay_steps", 1000))
    alpha        = float(hp.get("lr_alpha", 0.05))

    decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=peak_lr, decay_steps=decay_steps, alpha=alpha,
    )

    class _WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_lr = peak_lr * tf.clip_by_value(step / warmup_steps, 0.0, 1.0)
            post_warmup_lr = decay(step - warmup_steps)
            return tf.where(step < warmup_steps, warmup_lr, post_warmup_lr)

        def get_config(self):
            return {
                "peak_lr": peak_lr, "warmup_steps": warmup_steps,
                "decay_steps": decay_steps, "alpha": alpha,
            }

    return _WarmupCosineDecay()


def _make_learned_nu_model(base_model, rho_init: float, lam: float,
                            s_sse: float = 1.0, s_t: float = 1.0, name: str = "LSTM-SSE-t-Student",
                            grad_noise: bool = False, grad_noise_eta: float = 0.3,
                            grad_noise_gamma: float = 0.55,
                            lam_anneal: bool = False, lam_start: float = 0.1,
                            lam_end: float | None = None, lam_anneal_steps: int = 1000,
                            use_qlike: bool = False, s_qlike: float = 1.0,
                            nu_max: float | None = None):
    """
    Section 8: wraps a base model (predicting sigma2_t directly, via the
    base model's own activation="exponential" output layer) with a
    trainable, global Student-t ν = 2 + softplus(ρ), via a custom
    train_step/test_step so ν participates in gradient descent alongside
    the network's weights. predict()/call() are unchanged (same (batch, 1)
    sigma2 output as the fixed-ν path) — only the loss computation and
    optimizer step differ, which keeps every downstream
    model.predict(...) call site (sigma2_from_direct_output conversion,
    etc.) working exactly as it does for the fixed-ν path.

    grad_noise (2026-08-01, feasibility check): opt-in additive gradient
    noise (Neelakantan et al. 2015, "Adding Gradient Noise Improves
    Learning for Very Deep Networks") -- N(0, sigma_t^2) added to every
    trainable-variable gradient each step, with sigma_t = grad_noise_eta
    / (1+t)^grad_noise_gamma decaying over optimizer steps t
    (self.optimizer.iterations). Motivation: under full_batch_training
    (one deterministic gradient step per epoch), there is no mini-batch
    sampling noise to help escape a flat, near-constant-sigma2_t local
    optimum -- this restores an analogous source of stochasticity
    directly on the gradients themselves, unlike the adaptive_lr
    warm-up+cosine-decay schedule (which perturbs the step size, not the
    gradient direction, and per its own docstring was found to give no
    consistent collapse benefit).

    lam_anneal (2026-08-01, feasibility check): opt-in lambda annealing --
    instead of a fixed λ for the whole run, linearly ramp λ from
    lam_start to lam_end (default: the fixed `lam` argument) over
    lam_anneal_steps optimizer steps, then hold at lam_end. Analogous to
    KL-annealing / β-annealing in VAEs (Bowman et al. 2015), used there
    to fight "posterior collapse" -- the same qualitative failure mode as
    σ̂²_t collapsing to a near-constant value here: a loss term with a
    cheap degenerate optimum (KL there, the Student-t NLL here) can win
    before the rest of the network learns anything useful, unless its
    weight starts low and only ramps up once some real structure exists.
    λ is still never a trainable weight (no gradient flows into it) --
    only its *value*, read from a step-indexed schedule instead of a
    constant, changes.

    use_qlike (2026-08-01, feasibility check): if True, replaces the SSE
    term with QLIKE (see hybrid_loss_components_sigma2_tf's docstring)
    in the mixed loss, normalized by s_qlike instead of s_sse. Motivation:
    an outlier-dominance check found 82-99% of L_sse's per-epoch value
    came from the top 1% of observations across every series checked --
    i.e. L_sse was providing almost no gradient signal about ordinary
    day-to-day variation, just noise from a handful of extreme days.
    QLIKE (Patton 2011) is the standard robust alternative for volatility
    loss functions for exactly this reason.

    nu_max (2026-08-03, feasibility check): opt-in upper bound on the
    learned ν. Default None reproduces the original unbounded
    ν = 2 + softplus(ρ) (softplus has no ceiling, so nothing stops ν from
    drifting arbitrarily high -- i.e. arbitrarily close to Gaussian,
    which is the opposite of the heavy-tail hypothesis this whole
    Student-t setup exists to test). When nu_max is set, the
    reparameterization switches to ν = 2 + (nu_max - 2)·sigmoid(ρ), a
    bounded map onto (2, nu_max) -- ρ=0 lands at the midpoint of that
    range regardless of nu_max, matching this project's existing
    "neutral, non-circular" ν=5 starting convention when nu_max=8 (see
    src.eval.arch_restricted_recovery._neutral_nu_rho_init). This forces
    the model to commit to a heavier tail than it would otherwise choose,
    rather than adding a soft preference for one (a penalty term would
    still let the optimizer trade it off against the rest of the loss).

    A local class (not module-level) matching this file's lazy-TF-import
    convention: tf.keras.Model must already be resolvable when the class
    body evaluates.
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import hybrid_loss_components_sigma2_tf

    lam_start_val = float(np.clip(lam_start, 0.0, 1.0))
    lam_end_val = float(np.clip(lam_end, 0.0, 1.0)) if lam_end is not None else float(np.clip(lam, 0.0, 1.0))
    lam_anneal_steps_val = float(max(int(lam_anneal_steps), 1))
    nu_max_val = None if nu_max is None else float(nu_max)

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
            self.grad_noise = bool(grad_noise)
            self.grad_noise_eta = float(grad_noise_eta)
            self.grad_noise_gamma = float(grad_noise_gamma)
            self.lam_anneal = bool(lam_anneal)
            self.lam_start_val = lam_start_val
            self.lam_end_val = lam_end_val
            self.lam_anneal_steps_val = lam_anneal_steps_val
            self.use_qlike = bool(use_qlike)
            self.s_qlike_val = float(s_qlike)
            self.nu_max_val = nu_max_val
            # Keras 3's base Model.compile() creates its own internal
            # loss Mean tracker (present in self.metrics) regardless of
            # whether compile(loss=...) is passed. Our train_step/
            # test_step never touched it, so it stayed at its reset
            # value 0.0 forever -- and Keras 3's fit()/evaluate() epoch
            # logs are derived from self.metrics, NOT from whatever dict
            # a custom train_step returns, so history.history["loss"]/
            # ["val_loss"] (and therefore EarlyStopping, which monitors
            # "val_loss") silently saw a constant 0.0 every epoch despite
            # real, non-zero gradient updates actually happening. Fixed
            # per Keras's documented custom-train-step pattern: own the
            # tracker explicitly, update it with the real loss, and
            # expose ONLY it via the metrics property so Keras's
            # automatic per-epoch reset/reporting uses our real value.
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        @property
        def metrics(self):
            return [self.loss_tracker]

        def call(self, inputs, training=False):
            return self.base_model(inputs, training=training)

        def nu(self):
            if self.nu_max_val is None:
                return 2.0 + tf.nn.softplus(self.rho)
            return 2.0 + (self.nu_max_val - 2.0) * tf.sigmoid(self.rho)

        def _current_lam(self):
            step = tf.cast(self.optimizer.iterations, tf.float32)
            frac = tf.clip_by_value(step / self.lam_anneal_steps_val, 0.0, 1.0)
            return self.lam_start_val + (self.lam_end_val - self.lam_start_val) * frac

        def _compute_loss(self, y_true, y_pred):
            lam = self._current_lam() if self.lam_anneal else self.lam_val
            nu_t = self.nu()
            L_first, L_t = hybrid_loss_components_sigma2_tf(
                y_true, y_pred, nu_t, self.s_sse_val, self.s_t_val,
                use_qlike=self.use_qlike, s_qlike=self.s_qlike_val,
            )
            # hybrid_loss_components_sigma2_tf unconditionally drops the
            # Student-t normalizing constant c(nu) = -lgamma((nu+1)/2) +
            # lgamma(nu/2) + 0.5*log(pi*(nu-2)) -- harmless when nu is a
            # fixed float (c(nu) doesn't depend on any trainable weight,
            # so omitting it never changes a gradient), but THIS class
            # always has nu trainable (self.rho), and dc/dnu != 0 across
            # the whole training range (e.g. 0.057 at nu=5, 0.017 at
            # nu=8, verified via digamma) -- so minimizing the truncated
            # L_t w.r.t. nu is NOT equivalent to maximizing the full
            # likelihood; its gradient's fixed point differs from the
            # true MLE nu. Add the constant back only here (the fixed-nu
            # path never instantiates this class, so it's untouched).
            const = (-tf.math.lgamma(0.5 * (nu_t + 1.0)) + tf.math.lgamma(0.5 * nu_t)
                     + 0.5 * tf.math.log(np.pi * (nu_t - 2.0)))
            L_t = L_t + const / self.s_t_val
            return (1.0 - lam) * L_first + lam * L_t

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self._compute_loss(y, y_pred)
            trainable_vars = self.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            if self.grad_noise:
                step = tf.cast(self.optimizer.iterations, tf.float32)
                noise_std = self.grad_noise_eta / tf.pow(1.0 + step, self.grad_noise_gamma)
                grads = [
                    g + tf.random.normal(tf.shape(g), stddev=noise_std) if g is not None else g
                    for g in grads
                ]
            self.optimizer.apply_gradients(zip(grads, trainable_vars))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def test_step(self, data):
            x, y = data
            y_pred = self(x, training=False)
            loss = self._compute_loss(y, y_pred)
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

    return _LearnedNuModel(name=name)


def build_lstm_t_student(hp: dict) -> "tf.keras.Model":
    """
    LSTM trained with the hybrid (1−λ)·MSE/s_sse + λ·NLL_Student-t/s_t
    loss, optimized end-to-end by gradient-based maximum likelihood.

    By design this model uses NO GARCH parameters or GARCH-derived
    quantities anywhere -- not for weight injection, not for the initial
    cell state, not even to seed ν's starting value. Its input is ε²_t
    (the realized-variance proxy, scaled by src.data.scaling exactly as
    every other neural model here), and its weights are the ONLY thing
    that has to discover any GARCH-like dynamics; nothing is copied in.
    This is the intentional counterpoint to the (separate, still-intact)
    Section 6 src.models.garch_init verification path, which explicitly
    injects the GARCH solution into an LSTM cell to check whether the
    architecture CAN reproduce it -- that check is a structural-capacity
    diagnostic, not part of how this model is actually fit.

    Output layer: Dense(1, activation=None) emitting a pre-activation
    u_t, fed through a Lambda clipping u_t to [-20, 20] (Section 3's
    shared _LOG_VAR_CLIP, imported from hybrid_student_t) before exp().
    σ̂²_t > 0 is still guaranteed by construction (exp always positive),
    but now with the same overflow/underflow guard every other
    architecture's u_t=log(σ̂²_t) output already had -- this model was
    previously the only one without it, since activation="exponential"
    applies exp() straight to an unbounded pre-activation. That asymmetry
    was diagnosed as the most direct structural explanation for the
    sigma2 collapse observed on DJIA/SP500/GOLD/OIL (see
    logs/degeneracy.log): an unclipped exp() lets gradient descent push
    the pre-activation arbitrarily negative, saturating σ̂²_t at the
    float32-underflow floor with a vanishing local gradient and no
    numerical guard-rail to prevent it. This model's raw predict() output
    IS still σ̂²_t directly (see src.losses.hybrid_student_t.
    sigma2_from_direct_output and hybrid_loss_components_sigma2_tf) --
    only what happens between the linear layer and that output changed.

    Extra hp keys: nu (degrees of freedom, used when hp["nu_mode"] is
    not "learned"), lam (λ mixing weight), s_sse / s_t (frozen
    normalization scales — see src.losses.hybrid_student_t.
    compute_loss_scales; default 1.0/1.0 reproduces the un-normalized
    loss). adaptive_lr (bool, default False) swaps the constant
    hp["learning_rate"] for the warm-up + plain-cosine-decay schedule in
    _make_adaptive_lr_schedule (see its docstring); optional
    lr_warmup_steps / lr_decay_steps / lr_alpha keys tune that schedule.
    grad_noise (bool, default False; nu_mode="learned" only) adds decaying
    additive gradient noise, see _make_learned_nu_model's docstring;
    optional grad_noise_eta / grad_noise_gamma tune its schedule.
    lam_anneal (bool, default False; nu_mode="learned" only) linearly
    ramps λ from lam_start to lam_end over lam_anneal_steps optimizer
    steps instead of holding hp["lam"] fixed for the whole run -- see
    _make_learned_nu_model's docstring (KL-annealing analogy). lam_end
    defaults to hp["lam"] if not given.
    use_qlike (bool, default False; nu_mode="learned" only) replaces the
    SSE term with QLIKE (Patton 2011), normalized by s_qlike instead of
    s_sse -- see _make_learned_nu_model's docstring.

    hp["nu_mode"] (Section 8):
      "fixed" (default) — ν = hp["nu"], a constant (either sampled by
               the hyperparameter search, in which case it was
               previously selected by validation forecast loss -- see
               "likelihood_search" below for the corrected alternative
               -- or set directly by the caller).
      "learned" — ν = 2 + softplus(ρ) is a trainable model parameter,
               ρ initialized from hp["nu_rho_init"] (a fixed, neutral
               starting point chosen independently of any series' GARCH
               fit -- see tune_and_train._run_model_series). Returns a
               LearnedNuModel wrapping the usual functional model;
               predict() output is unchanged.
      Any other value of hp["nu"] set by an outer "likelihood_search"
      selection (Section 8) is just a fixed ν chosen by OOS likelihood
      rather than validation MSE/hybrid-loss -- from this function's
      point of view that is identical to "fixed" (nu_mode stays "fixed"
      or is omitted; the search happens one level up, in
      src.tuning.tune_and_train).
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_hybrid_loss_sigma2, _LOG_VAR_CLIP

    units    = hp["lstm_units"]
    drop     = hp.get("dropout", 0.0)
    lam      = hp["lam"]
    W        = hp["window_size"]
    nu_mode  = hp.get("nu_mode", "fixed")
    s_sse = hp.get("s_sse", 1.0)
    s_t   = hp.get("s_t", 1.0)

    lr_or_schedule = (
        _make_adaptive_lr_schedule(hp) if hp.get("adaptive_lr", False)
        else hp.get("learning_rate", 1e-3)
    )

    inp = tf.keras.Input(shape=(W, 1), name="eps_window")
    x   = tf.keras.layers.LSTM(units, dropout=drop, recurrent_dropout=0.0,
                                kernel_initializer="glorot_uniform")(inp)
    x   = tf.keras.layers.Dropout(drop)(x)
    u   = tf.keras.layers.Dense(1, activation=None, name="log_sigma2_pre")(x)
    out = tf.keras.layers.Lambda(
        lambda t: tf.exp(tf.clip_by_value(t, -_LOG_VAR_CLIP, _LOG_VAR_CLIP)),
        name="sigma2",
    )(u)

    base_model = tf.keras.Model(inp, out, name="LSTM-SSE-t-Student")

    if nu_mode == "learned":
        model = _make_learned_nu_model(
            base_model, rho_init=hp["nu_rho_init"], lam=lam, s_sse=s_sse, s_t=s_t,
            name="LSTM-SSE-t-Student",
            grad_noise=hp.get("grad_noise", False),
            grad_noise_eta=hp.get("grad_noise_eta", 0.3),
            grad_noise_gamma=hp.get("grad_noise_gamma", 0.55),
            lam_anneal=hp.get("lam_anneal", False),
            lam_start=hp.get("lam_start", 0.1),
            lam_end=hp.get("lam_end"),
            lam_anneal_steps=hp.get("lam_anneal_steps", 1000),
            use_qlike=hp.get("use_qlike", False),
            s_qlike=hp.get("s_qlike", 1.0),
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr_or_schedule))
    else:
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_or_schedule),
            loss=make_hybrid_loss_sigma2(nu=hp["nu"], lam=lam, s_sse=s_sse, s_t=s_t),
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
