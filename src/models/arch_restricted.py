"""
arch_restricted.py — ARCH(1)-restricted LSTM: an optimizer diagnostic,
not a new claim about the theory.

Motivation
----------
The LSTM cell-state recursion already has the algebraic shape of a GARCH
recursion:

    c_t = f_t · c_{t-1} + i_t · c̃_t        (LSTM)
    σ²_t = β  · σ²_{t-1} + α₁ · ε²_{t-1}    (GARCH)

Five structural constraints collapse the general cell down to ARCH(1)
exactly:

  1. Forget gate closed:   b_f = -30  ⟹  f_t ≈ 0.  ARCH(1) has no β term.
     (Making b_f trainable instead — i.e. b_f = logit(β) — is the natural
     extension to GARCH(1,1); see `forget_gate_trainable` below.)
  2. Input gate saturated open: b_i = +30  ⟹  i_t ≈ 1.
  3. Linear activation on the candidate: c̃_t = W_c·x_t + b_c, unbounded
     (tanh would clip c̃_t at ±1 and could never represent ω + α₁ε²).
  4. No recurrent connections anywhere (U_i = U_f = U_c = U_o = 0): gates
     and candidate depend only on x_t, never on h_{t-1} — otherwise this
     stops being ARCH and becomes something state-dependent in a way
     ARCH(1) is not.
  5. Output gate saturated open: b_o = +30  ⟹  h_t ≈ c_t.

With all five in place the cell collapses to exactly
    h_t = ω̂ + α̂₁·ε²_{t-1},
with W_c = α̂₁ = softplus(alpha_raw), b_c = ω̂ = softplus(omega_raw) the
only two free trainable scalars (alpha_raw/omega_raw, unconstrained; the
softplus keeps α̂₁, ω̂ >= 0 by construction — positivity is then structural,
not a downstream clip, see _RestrictedGateLSTMCell's docstring). Training
this restricted network under pure Student-t maximum likelihood (λ=1, see
`hp["lam"]` below) and comparing the recovered (ω̂, α̂₁, ν̂) against
`arch`'s own ARCH(1)-t MLE (outputs/models/ARCH1/<series>/params.json)
is a pure optimizer diagnostic: it says nothing about whether the
LSTM-t-Student architecture is theoretically capable of representing
ARCH(1) (src.models.garch_init already proves the analogous GARCH(1,1)
case constructively, by injecting the solution rather than searching for
it) — only whether gradient-based training, run through the exact same
pipeline (Adam, the project's LR schedule / grad-noise / learned-ν
machinery) used for the real proposed model, actually FINDS that known
optimum when the architecture has no slack capacity left to hide a
degenerate solution in.

Why a hand-written Layer, not tf.keras.layers.LSTM
----------------------------------------------------
Keras' built-in LSTM exposes kernel/recurrent_kernel/bias as monolithic
trainable tensors — there is no way to mark some scalar entries
trainable and others permanently fixed within the same tensor. A
minimal explicit single-unit cell is the direct way to implement
"i_t, o_t frozen open; f_t frozen closed (or trainable, for GARCH); no
recurrent connections; only the candidate kernel/bias trainable" exactly
as specified. Same reasoning as src.models.ablation_ladder._Rung1Cell,
which solves the analogous GARCH(1,1) recovery problem with a
hand-written *whole-series* recursion; this module instead follows the
*windowed* convention every other neural model in this project uses
(src.models.neural, src.tuning.tune_and_train._make_windows) so training
runs through the identical Adam/EarlyStopping/multi-seed pipeline as the
real proposed model — that pipeline parity is the entire point of the
diagnostic. Because f_t ≈ 0 (ARCH has no memory beyond one lag), the
window boundary is not a limitation here: after the very first step
within a window the carried-over state is already discarded, so any
window length W ≥ 1 recovers the same answer.

Section-4 input scaling
------------------------
Every neural model in this project consumes x_t = ε²_t / σ̄²_train (the
scaled proxy, src.data.scaling), never raw ε²_t — required for
Proposition 2 gate correspondence to be structurally meaningful and kept
here for the same reason (and so this model plugs into the existing
windowing/data-loading code unchanged). The cell folds the
`sigma2_train_scaler` back in internally so its two trainable scalars
(`alpha_hat`, `omega_hat`) come out directly in `arch`'s own units — no
post-hoc rescaling needed when comparing against params.json.

Comparability with arch's own MLE
-----------------------------------
ARCH(1) in `arch` (src.models.econometric.ARCH1Model) is fit by maximum
likelihood. For the comparison in `src.eval.arch_restricted_recovery` to
be apples-to-apples (same estimator, not "MLE vs. some other objective"),
this architecture MUST be trained with λ=1 (pure Student-t NLL, no SSE/
QLIKE term) — hp["lam"] is accepted like any other hybrid-loss hp key
(no special-casing needed: at λ=1 the first loss term's weight is
exactly 0), but it is the CALLER's responsibility to actually pass 1.0;
this module does not enforce it, since diagnostics at other λ values may
be independently useful.
"""
from __future__ import annotations

import numpy as np


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _inv_softplus(y: float) -> float:
    """rho such that softplus(rho) = y (y clamped >= 1e-6 -- softplus(x) > 0 for every
    finite x, so y=0 exactly has no finite preimage; this reproduces omega_init=0.0's
    "near-zero, neutral" intent instead)."""
    y = max(float(y), 1e-6)
    return float(np.log(np.expm1(y)))


# ──────────────────────────────────────────────────────────────────────────────
# The restricted cell
# ──────────────────────────────────────────────────────────────────────────────

def _build_restricted_cell_class():
    """
    Returns the _RestrictedGateLSTMCell class. Factored into a function
    (not a module-level class) so tf.keras.layers.Layer is only resolved
    once TensorFlow is actually imported, matching this project's
    lazy-TF-import convention (see e.g. src.models.garch_init).
    """
    import tensorflow as tf

    class _RestrictedGateLSTMCell(tf.keras.layers.Layer):
        """
        Single-unit cell implementing the 5-constraint ARCH(1)/GARCH(1,1)
        reduction described in this module's docstring.

        Trainable weights
        ------------------
        alpha_hat = softplus(alpha_raw) >= 0, omega_hat = softplus(omega_raw)
            >= 0 -- alpha_raw/omega_raw are the actual tf.Variables (any real
            number); alpha_hat/omega_hat are read-only properties, still
            directly comparable to arch's own (alpha[1], omega) since
            softplus only restricts the codomain to (0, inf), it does not
            rescale/reparametrize the recursion's units otherwise (see
            "Positivity" below).
        beta_bias : only created (and trainable) if forget_gate_trainable
            — sigmoid(beta_bias) = beta_hat, the GARCH(1,1) extension.

        Everything else (i_t's and o_t's bias, every gate/candidate
        kernel and recurrent-kernel) is a fixed tf.constant, never a
        tf.Variable — "fixed" here means genuinely absent from
        trainable_variables, not merely initialized-then-never-updated.

        Positivity (structural, not a downstream clip)
        -------------------------------------------------
        h_t = o_t*c_t is a sum of non-negative terms by induction: c_0 = 0;
        alpha_hat, omega_hat >= 0 (softplus); x_t = eps2_t/sigma2_train >= 0
        (a scaled square); i_t, o_t, f_t in (0, 1) (sigmoid) times a
        non-negative c_{t-1} stays non-negative. So sigma2_t >= 0 for EVERY
        input, no output-side clip/softplus needed (an earlier sigma2_floor
        opt-in did h_T = floor + softplus(h_T_raw), wrapping the whole
        linear combination -- that also guarantees positivity, but makes
        d(sigma2_t)/d(eps2_{t-1}) = alpha_hat * softplus'(h_T_raw), a
        state-dependent quantity, not the constant alpha_hat GARCH's own
        alpha[1] is. Constraining alpha_hat/omega_hat individually instead
        keeps the marginal sensitivity exactly alpha_hat, matching the
        "alpha <-> d(sigma2_t)/d(eps2_{t-1})" reading this diagnostic
        depends on). Only alpha_hat, omega_hat, and f_t/i_t (already
        sigmoid-bounded, unchanged) needed a constraint; the induction goes
        through without ever touching h_t directly.
        """

        def __init__(self, sigma2_train_scaler: float, forget_gate_trainable: bool = False,
                     beta_init: float = 0.85, alpha_init: float = 0.05, omega_init: float = 0.0,
                     persistence_max: float = 0.999,
                     gate_saturation_bias: float = 30.0,
                     name: str = "restricted_gate_lstm_cell", **kwargs):
            super().__init__(name=name, **kwargs)
            self.sigma2_train_scaler = float(sigma2_train_scaler)
            self.forget_gate_trainable = bool(forget_gate_trainable)
            self._beta_init = float(beta_init)
            self._alpha_init = float(alpha_init)
            self._omega_init = float(omega_init)
            self._persistence_max = float(persistence_max)
            self._gate_bias = float(gate_saturation_bias)

        def build(self, input_shape):
            # trainable=False when forget_gate_trainable=True: alpha_hat is
            # derived from persistence_raw/mix_raw in that branch (see
            # below), so alpha_raw would otherwise sit in
            # trainable_variables with no path to the loss -- harmless
            # (Adam skips None-gradient vars) but prints a
            # "Gradients do not exist" warning on every step.
            self.alpha_raw = self.add_weight(
                name="alpha_raw", shape=(), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(_inv_softplus(self._alpha_init)),
                trainable=(not self.forget_gate_trainable),
            )
            self.omega_raw = self.add_weight(
                name="omega_raw", shape=(), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(_inv_softplus(self._omega_init)),
                trainable=True,
            )
            if self.forget_gate_trainable:
                # Persistence/mix reparametrization (replaces independent
                # alpha_hat/beta_hat for this branch only): alpha_hat and
                # beta_hat share a single bounded "budget"
                # persistence_hat = persistence_max*sigmoid(persistence_raw)
                # in (0, persistence_max), split between them by
                # mix_raw. This makes alpha_hat + beta_hat < persistence_max
                # a STRUCTURAL guarantee (stationarity), not something the
                # optimizer has to discover on its own -- the prior
                # independent sigmoid(beta_bias)/softplus(alpha_raw)
                # parametrization let both grow large simultaneously, and
                # the GARCH(1,1)-restricted recovery diagnostic found
                # alpha_hat+beta_hat > 1 (non-stationary) in all 6 series.
                pers_init = min(self._alpha_init + self._beta_init, 0.995 * self._persistence_max)
                mix_init = self._beta_init / pers_init if pers_init > 0 else 0.5
                self.persistence_raw = self.add_weight(
                    name="persistence_raw", shape=(), dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(_logit(pers_init / self._persistence_max)),
                    trainable=True,
                )
                self.mix_raw = self.add_weight(
                    name="mix_raw", shape=(), dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(_logit(mix_init)),
                    trainable=True,
                )
            # b_i, b_o: fixed constants, never trainable — bullets 2 and 5.
            self._b_i = tf.constant(self._gate_bias, dtype=tf.float32)
            self._b_o = tf.constant(self._gate_bias, dtype=tf.float32)
            super().build(input_shape)

        @property
        def persistence_hat(self):
            return self._persistence_max * tf.sigmoid(self.persistence_raw)

        @property
        def alpha_hat(self):
            if self.forget_gate_trainable:
                return self.persistence_hat * (1.0 - tf.sigmoid(self.mix_raw))
            return tf.nn.softplus(self.alpha_raw)   # ARCH(1) unchanged

        @property
        def beta_hat(self):
            if self.forget_gate_trainable:
                return self.persistence_hat * tf.sigmoid(self.mix_raw)
            return tf.constant(0.0, dtype=tf.float32)   # ARCH(1): forget gate fixed closed

        @property
        def omega_hat(self):
            return tf.nn.softplus(self.omega_raw)

        def call(self, inputs):
            # inputs: (batch, W, 1), Section-4-scaled x_t = eps2_t / sigma2_train.
            x = tf.squeeze(tf.cast(inputs, tf.float32), axis=-1)   # (batch, W)
            x_time_major = tf.transpose(x)                          # (W, batch) — scan over time

            i_t = tf.sigmoid(self._b_i)         # ~= 1, no kernel/recurrent dependence
            f_t = self.beta_hat                  # ~= 0 for ARCH(1); trainable (persistence-bounded) for GARCH(1,1)
            o_t = tf.sigmoid(self._b_o)          # ~= 1

            def step(c_prev, x_t):
                # Linear candidate, no recurrent connection (bullets 3+4).
                # alpha_hat/omega_hat (softplus-constrained >= 0) are already
                # in arch's own units; sigma2_train_scaler undoes the
                # Section-4 input scaling.
                g_t = self.alpha_hat * self.sigma2_train_scaler * x_t + self.omega_hat
                return f_t * c_prev + i_t * g_t

            c0 = tf.zeros_like(x_time_major[0])                     # (batch,) fresh per-window state, >= 0
            c_seq = tf.scan(step, x_time_major, initializer=c0)      # (W, batch), >= 0 by induction (see class docstring)
            c_T = c_seq[-1]                                          # (batch,)
            h_T = o_t * c_T                                          # bullet 5: h_t ~= c_t; >= 0, no clip needed
            return tf.expand_dims(h_T, axis=-1)                      # (batch, 1) — sigma2 output convention

        def get_config(self):
            config = super().get_config()
            config.update({
                "sigma2_train_scaler": self.sigma2_train_scaler,
                "forget_gate_trainable": self.forget_gate_trainable,
            })
            return config

    return _RestrictedGateLSTMCell


# ──────────────────────────────────────────────────────────────────────────────
# Model builder — mirrors src.models.neural.build_lstm_t_student's
# signature/conventions so it plugs into the same training infra
# (src.tuning.tune_and_train._train_one / _multiseed_train_and_predict)
# with no duplication of the Adam/loss/nu-handling machinery.
# ──────────────────────────────────────────────────────────────────────────────

def build_arch_restricted_lstm(hp: dict) -> "tf.keras.Model":
    """
    ARCH(1)-restricted (or, with hp["forget_gate_trainable"]=True,
    GARCH(1,1)-restricted) LSTM. Reuses build_lstm_t_student's own Adam
    optimizer / adaptive-LR schedule / learned-or-fixed-ν wrapper
    (src.models.neural._make_learned_nu_model, imported directly) and the
    project's hybrid Student-t loss (src.losses.hybrid_student_t) — the
    ONLY thing that differs from the real proposed model is the
    architecture between the input window and the sigma2 output: a
    single _RestrictedGateLSTMCell instead of a free tf.keras.layers.LSTM
    + Dense head. See module docstring for the full derivation.

    Required hp keys
    -----------------
    window_size            : int, same convention as every other builder.
    sigma2_train_scaler     : float, this series' scaler.json
                              ["sigma2_train"] value (Section 4) — needed
                              to undo the input scaling inside the cell.
    lam                     : float in [0, 1]. MUST be 1.0 for the
                              recovered (omega_hat, alpha_hat, nu_hat) to
                              be comparable to arch's own MLE (see module
                              docstring) — not enforced here, only by the
                              caller (src.eval.arch_restricted_recovery).

    Optional hp keys
    ------------------
    forget_gate_trainable   : bool, default False. False = ARCH(1)
                              (b_f fixed at -30). True = GARCH(1,1) (b_f
                              trainable, "the natural extension").
    beta_init               : float, default 0.85. Neutral starting point
                              for beta_hat when forget_gate_trainable.
    alpha_init / omega_init : float, defaults 0.05 / 0.0. Neutral
                              starting points for alpha_hat / omega_hat —
                              NOT tied to any series' actual ARCH(1)/
                              GARCH(1,1) answer, so recovery at
                              convergence is genuine, not circular (same
                              principle as
                              src.models.ablation_ladder._NEUTRAL_*_INIT).
    persistence_max          : float, default 0.999, forget_gate_trainable
                              only. Hard upper bound on alpha_hat+beta_hat
                              (persistence/mix reparametrization -- see
                              _RestrictedGateLSTMCell.build) that makes
                              stationarity a structural guarantee instead
                              of something the optimizer has to discover;
                              the prior independent alpha/beta
                              parametrization let both grow large at once
                              (all 6 series recovered alpha_hat+beta_hat
                              > 1, non-stationary, in the pre-fix
                              diagnostic).
    nu_mode                 : "fixed" (hp["nu"] constant) or "learned"
                              (nu = 2 + softplus(rho), hp["nu_rho_init"]
                              required) — identical semantics to
                              build_lstm_t_student.
    learning_rate / adaptive_lr (+ lr_warmup_steps / lr_decay_steps /
    lr_alpha), grad_noise (+ eta / gamma), s_sse / s_t : identical
    semantics/defaults to build_lstm_t_student.
    use_qlike / s_qlike     : bool (default False) / float (default 1.0),
                              nu_mode="learned" only. Replaces the L_first
                              term with QLIKE (Patton 2011) instead of
                              SSE — see hybrid_loss_components_sigma2_tf's
                              docstring. Only meaningful when lam < 1
                              (L_first's weight is (1-lam)); wire this up
                              for a lam-sweep that compares SSE- vs
                              QLIKE-anchored fits, not for the lam=1.0
                              recovery diagnostic itself, where L_first's
                              weight is exactly 0 regardless.
    nu_max                  : float, default None (unchanged unbounded
                              ν = 2 + softplus(ρ)), nu_mode="learned" only.
                              When set, ν = 2 + (nu_max-2)·sigmoid(ρ) is
                              bounded onto (2, nu_max) — see
                              _make_learned_nu_model's docstring. Use to
                              force the model to commit to heavier tails
                              (lower ν ceiling) than it would otherwise
                              settle on.
    """
    import tensorflow as tf
    from src.losses.hybrid_student_t import make_hybrid_loss_sigma2
    from src.models.neural import _make_learned_nu_model, _make_adaptive_lr_schedule

    W = hp["window_size"]
    lam = hp["lam"]
    nu_mode = hp.get("nu_mode", "fixed")
    s_sse = hp.get("s_sse", 1.0)
    s_t = hp.get("s_t", 1.0)
    sigma2_train_scaler = hp["sigma2_train_scaler"]
    forget_gate_trainable = bool(hp.get("forget_gate_trainable", False))
    beta_init = hp.get("beta_init", 0.85)
    alpha_init = hp.get("alpha_init", 0.05)
    omega_init = hp.get("omega_init", 0.0)
    persistence_max = hp.get("persistence_max", 0.999)

    lr_or_schedule = (
        _make_adaptive_lr_schedule(hp) if hp.get("adaptive_lr", False)
        else hp.get("learning_rate", 1e-3)
    )

    RestrictedCell = _build_restricted_cell_class()

    model_name = "GARCH11-Restricted-LSTM" if forget_gate_trainable else "ARCH-LSTM"

    inp = tf.keras.Input(shape=(W, 1), name="eps_window_scaled")
    out = RestrictedCell(
        sigma2_train_scaler=sigma2_train_scaler,
        forget_gate_trainable=forget_gate_trainable,
        beta_init=beta_init, alpha_init=alpha_init, omega_init=omega_init,
        persistence_max=persistence_max,
        name="restricted_cell",
    )(inp)

    base_model = tf.keras.Model(inp, out, name=model_name)

    if nu_mode == "learned":
        model = _make_learned_nu_model(
            base_model, rho_init=hp["nu_rho_init"], lam=lam, s_sse=s_sse, s_t=s_t,
            name=model_name,
            grad_noise=hp.get("grad_noise", False),
            grad_noise_eta=hp.get("grad_noise_eta", 0.3),
            grad_noise_gamma=hp.get("grad_noise_gamma", 0.55),
            use_qlike=hp.get("use_qlike", False),
            s_qlike=hp.get("s_qlike", 1.0),
            nu_max=hp.get("nu_max", None),
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr_or_schedule))
    else:
        model = base_model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_or_schedule),
            loss=make_hybrid_loss_sigma2(nu=hp["nu"], lam=lam, s_sse=s_sse, s_t=s_t),
        )
    return model


def extract_arch_params(model) -> dict:
    """
    Read back the trained (alpha_hat, omega_hat, beta_hat, nu_hat) from a
    model built by build_arch_restricted_lstm — already in arch's own
    units (see _RestrictedGateLSTMCell), no further rescaling needed.
    beta_hat is 0.0 when the model was built with forget_gate_trainable
    =False (ARCH(1)); nu_hat is None if the model was built with
    nu_mode="fixed" (nothing was learned, hp["nu"] is already known to
    the caller).
    """
    import tensorflow as tf

    base = model.base_model if hasattr(model, "base_model") else model
    cell = None
    for layer in base.layers:
        if "restricted_gate_lstm_cell" in layer.name or layer.name == "restricted_cell":
            cell = layer
            break
    if cell is None:
        raise ValueError("No restricted-gate cell layer found in model.")

    alpha_hat = float(cell.alpha_hat.numpy())
    omega_hat = float(cell.omega_hat.numpy())
    beta_hat = float(cell.beta_hat.numpy())
    nu_hat = float(model.nu().numpy()) if hasattr(model, "nu") else None

    return {"alpha_hat": alpha_hat, "omega_hat": omega_hat, "beta_hat": beta_hat, "nu_hat": nu_hat}
