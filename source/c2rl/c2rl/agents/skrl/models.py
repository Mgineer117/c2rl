"""skrl-compatible model wrappers for c2rl actors and critics.

Observation layout: ``s = {x, xrefs, urefs}`` — see ``ref_window.py``. Every
model here recovers its own input shape from the Dict observation space via
``RefWindow.from_space``; none of them infers a layout from ``obs_dim``, which
is what the old ``_preview_width`` / ``obs_dim/2`` parity guessing did.

Actor   u = urefs[0] + W2(xrefs) @ tanh(W1(x) @ e)            (nn_modules.CLActor)
critic  v = MLP([phi(x, e) || psi(xrefs)])                    (RefWindowValueModel)

Both route the reference window through ``PreviewSequenceEncoder`` (mlp | gru |
attn, selected by ``--encoder``) and both see it through ``Feats.sequence``, so
neither ever reads an absolute position or a raw wrapping angle.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
except ImportError:
    raise ImportError("skrl is required. Install it or use the local developer copy.")

from .math_utils import rescale_residual
from .nn_modules import MLP, BoundedCCM_Generator, CCM_Generator, CLActor, PreviewSequenceEncoder
from .ref_window import Feats, RefWindow
from .state_symmetry import StateSymmetry

_MIN_LOG_STD = math.log(0.001)  # ≈ -6.908; matches CLActor annealing floor


def _layout(observation_space, x_dim=None, angle_idx=None, sym=None, ref_offset=1):
    """``(window, feats)`` for a model, from its Dict observation space.

    ``x_dim`` is accepted only to cross-Check what the space declares — a
    mismatch means the env and the runner disagree about the state layout,
    which used to be an entirely silent mis-slicing.
    """
    window = RefWindow.from_space(observation_space, offset=ref_offset)
    if x_dim is not None and int(x_dim) != window.x_dim:
        raise ValueError(
            f"x_dim mismatch: caller says {int(x_dim)}, observation space declares "
            f"{window.x_dim}. The env and the model disagree about the state layout.")
    return window, Feats(window.x_dim, angle_idx or [], sym)


class _AnalyticPotentialMixin:
    """Critic parameterization V(s) = ||e||^2_M + f_theta(s)  (O6).

    The Mahalanobis reward is the decrement form r_t = Phi(s_t+1) - Phi(s_t),
    Phi(s) = -||e||^2_M. Telescoping gives V_shaped = (1-gamma) V_orig - Phi, so
    the decrement form makes the reward o(dt) but does not remove the O(1)
    potential — it moves it into the critic's regression target, where
    -Phi(s) = ||e||^2_M is analytically known and dominates what the network must
    fit. Adding the closed form back lets f_theta represent only the genuinely
    O(dt) part. (Classical shaping/value-init equivalence, Wiewiora 2003.)

    Needs the frozen Phase-A CMG (attached by C2RLAgent post-synthesis); until
    then the term is absent and the model behaves as a plain critic.

    Scale: the added term is in real value units, so this is consistent only with
    the value preprocessor off (``use_value_norm=false``).
    """

    def _init_potential(self, window, feats, w_lb):
        self._pot_ccm_gen = None          # set by C2RLAgent._attach_critic_potential
        self._pot_window = window
        self._pot_feats = feats
        self._pot_w_lb = float(w_lb)

    def _analytic_potential(self, inputs):
        """``||e||^2_M`` from the observation, where ``e = x - xrefs[0]``."""
        gen = getattr(self, "_pot_ccm_gen", None)
        obs = inputs.get("observations")
        if gen is None or obs is None:
            return None
        from .math_utils import bound_W, spd_inverse
        x, xrefs, _ = self._pot_window.split(obs)
        with torch.no_grad():
            e = self._pot_feats.error(x, xrefs[:, 0]).unsqueeze(-1)
            W = bound_W(gen(x)[0], self._pot_w_lb, self._pot_window.x_dim,
                        getattr(gen, "bounded", False))
            M = spd_inverse(W)
            return torch.bmm(torch.bmm(e.transpose(1, 2), M), e).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────── #
# Critic
# ─────────────────────────────────────────────────────────────────────────── #

class RefWindowValueModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
    """Value/Q critic over ``{x, xrefs, urefs}``, structured to mirror the actor.

        phi = MLP([single(x), e])            state path      — mirrors W1(x)·e
        psi = Encoder(sequence(x, xrefs))    reference path  — mirrors W2(xrefs)
        V   = MLP([phi || psi])              combine

    The actor's ``W2(xrefs)`` emits a matrix that multiplies the error latent;
    a critic must emit a scalar, so the two embeddings meet in a joint MLP
    instead. ``phi`` sees both the invariant configuration of ``x`` (current
    speed etc. genuinely change the value beyond the error alone) and the
    tracking error; ``psi`` sees the reference window only, each point relative
    to the current ``x`` with wrapped angles.

    ``use_actions=True`` appends the action, making this a Q-function (SAC).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        network: list | None = None,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        use_actions: bool = False,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        # combine: "concat" only. See the commented-out "bilinear" (UVFA
        # factorization) and "film" (psi gates phi, the closest analogue of the
        # actor's gain modulation) at the end of this file.
        combine: str = "concat",
        analytic_potential: bool = False,
        w_lb: float = 0.01,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        net_spec = (network or [{}])[0] if network else {}
        hidden_dim = hidden_dim or net_spec.get("layers", [256, 256])
        activation = net_spec.get("activations", activation)
        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation

        if combine != "concat":
            raise ValueError(
                f"RefWindowValueModel: only combine='concat' is active (got {combine!r}); "
                f"'bilinear'/'film' are commented out at the end of models.py.")
        self.combine = combine

        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        act_dim = int(self.action_space.shape[0])
        self._use_actions = use_actions

        self.phi = MLP(self.feats.single_dim + self.feats.error_dim, list(hidden_dim),
                       encoder_hidden, activation=act_module)
        self.psi = PreviewSequenceEncoder(
            preview_dim=self.window.length * self.feats.pair_dim,
            point_dim=self.feats.pair_dim,
            hidden=encoder_hidden, out_dim=encoder_hidden,
            mode=encoder, stride=encoder_stride)
        self.net = MLP(2 * encoder_hidden + (act_dim if use_actions else 0),
                       list(hidden_dim), 1, activation=act_module)

        self.use_analytic_potential = bool(analytic_potential)
        self._init_potential(self.window, self.feats, w_lb)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "value"):
        obs = inputs["observations"]
        n = obs.shape[0]
        x, xrefs, _ = self.window.split(obs)
        e = self.feats.error(x, xrefs[:, 0])
        phi = self.phi(torch.cat([self.feats.single(x), e], dim=-1))
        psi = self.psi(self.feats.sequence(x, xrefs).reshape(n, -1))
        parts = [phi, psi]
        if self._use_actions:
            parts.append(inputs["taken_actions"])
        value = self.net(torch.cat(parts, dim=-1))
        if self.use_analytic_potential:
            pot = self._analytic_potential(inputs)
            if pot is not None:
                value = value + pot        # V = f_theta(s) + ||e||^2_M  (= -Phi(s))
        return value, {}


# Back-compat aliases: the runner/config layer still refers to the critic by its
# old names. One class serves both roles now (the privileged `states` channel is
# gone — the critic reads the same observation as the actor).
EmbeddedDeterministicModel = RefWindowValueModel
TrajectoryAwareValueModel = RefWindowValueModel


# ─────────────────────────────────────────────────────────────────────────── #
# Metric generator
# ─────────────────────────────────────────────────────────────────────────── #

class MetricModel(Model):
    """CCM_Generator wrapped as a skrl Model for checkpointing.

    The underlying CCM_Generator is accessed via ``self.ccm_gen`` by C3MAgent
    and C2RLAgent for the contraction loss computation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mode: str = "deterministic",
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        **kwargs,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space, device=device)
        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        self.x_dim, self.u_dim = self.window.x_dim, self.window.u_dim

        common = dict(x_dim=self.x_dim, hidden_dim=hidden_dim or [256, 256],
                      activation=activation, mode=mode,
                      device=str(device) if not isinstance(device, str) else device,
                      angle_idx=list(angle_idx or []), sym=sym)
        if kwargs.get("constrain_eigenvalues", False):
            # outputs_metric: cmg_method="cvstem" emits M directly (its SDP
            # targets can be inverted once offline, and the reward wants M), so
            # every env step drops a batched SPD inverse. "ccm" emits W, since
            # its C1/C2 losses are written in W and there is no dataset to
            # pre-invert. See BoundedCCM_Generator.
            self.ccm_gen = BoundedCCM_Generator(
                w_lb=kwargs.get("w_lb", 0.1), w_ub=kwargs.get("w_ub", 10.0),
                outputs_metric=bool(kwargs.get("outputs_metric", False)), **common)
        else:
            self.ccm_gen = CCM_Generator(**common)

    def compute(self, inputs: dict, role: str = "cmg"):
        x, _, _ = self.window.split(inputs["observations"])
        W, info = self.ccm_gen(x)
        return W.reshape(W.shape[0], -1), {}

    def act(self, inputs: dict, role: str = "cmg"):
        output, extra = self.compute(inputs, role)
        return output, None, extra

    def forward(self, inputs: dict, role: str = "cmg"):
        output, _ = self.compute(inputs, role)
        return output


# ─────────────────────────────────────────────────────────────────────────── #
# Actors — control backbone (CLActor)
# ─────────────────────────────────────────────────────────────────────────── #

def _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                    hidden_dim, activation, anneal_stddev,
                    encoder, encoder_hidden, encoder_stride):
    window, feats = _layout(observation_space, x_dim, angle_idx, sym)
    return CLActor(window=window, feats=feats, anneal_stddev=anneal_stddev,
                   hidden_dim=hidden_dim or [128, 128], activation=activation,
                   encoder=encoder, encoder_hidden=encoder_hidden,
                   encoder_stride=encoder_stride)


class CLDeterministicActorModel(DeterministicMixin, Model):
    """CLActor as a skrl Deterministic policy — C3M's default policy class
    (``class: DeterministicMixin`` in each ``skrl_c3m_cfg.yaml``). Unbounded;
    see ``SquashedCLDeterministicActorModel`` for ``backbone: control-squashed``.
    """

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False,
                 hidden_dim: list | None = None, activation: str = "tanh",
                 x_dim: int | None = None, angle_idx: list | None = None,
                 sym: StateSymmetry | None = None, encoder: str = "mlp",
                 encoder_hidden: int = 64, encoder_stride: int = 1, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation, False,
                                        encoder, encoder_hidden, encoder_stride)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        return self.cl_actor.mean_control(inputs["observations"]), {}


class SquashedCLDeterministicActorModel(DeterministicMixin, Model):
    """Tanh-squashed CLActor, deterministic — ``backbone: control-squashed`` for C3M.

    ``compute()`` returns ``cl_actor.mean_control_squashed`` (feedback
    tanh-bounded into the action space, ``urefs[0]`` added after squashing).

    C3M needs this rather than a hard ``clip_actions`` clamp: ``torch.clamp``'s
    gradient is exactly zero at saturation, so ``K = jacobian(u, x)`` collapses
    there too and the certificate silently degrades to checking open-Loop drift
    in exactly the states that most need feedback — a confirmed real AUC
    divergence. ``tanh``'s gradient never hits exact zero.
    """

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False,
                 hidden_dim: list | None = None, activation: str = "tanh",
                 x_dim: int | None = None, angle_idx: list | None = None,
                 sym: StateSymmetry | None = None, encoder: str = "mlp",
                 encoder_hidden: int = 64, encoder_stride: int = 1, **kwargs):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation, False,
                                        encoder, encoder_hidden, encoder_stride)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        return self.cl_actor.mean_control_squashed(
            inputs["observations"], self._d_min_actions, self._d_max_actions), {}


class CLActorModel(GaussianMixin, Model):
    """CLActor as a skrl Gaussian policy — ``backbone: control``.

    ``u = urefs[0] + W2(xrefs) @ tanh(W1(x) @ e)``. Unbounded; see
    ``SquashedCLActorModel`` for ``backbone: control-squashed``.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = _MIN_LOG_STD,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        anneal_stddev: bool = True,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        # anneal_stddev=False -> logstd is a learned parameter (requires_grad=True)
        # and CLActor.anneal_stddev() becomes inert, so nothing overwrites what
        # the policy gradient learns. True (default) is the frozen/scheduled path.
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation, anneal_stddev,
                                        encoder, encoder_hidden, encoder_stride)
        self.log_std_parameter = self.cl_actor.logstd

        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        state = inputs["observations"]
        # u = urefs[0] + pi(s); the analytic-base warm-start has been removed.
        feedback, uref = self.cl_actor._feedback(state)
        return uref + feedback, {"log_std": self.log_std_parameter}


# ─────────────────────────────────────────────────────────────────────────── #
# Actors — plain-MLP backbone
# ─────────────────────────────────────────────────────────────────────────── #

class MLPResidualActorModel(GaussianMixin, Model):
    """``backbone: mlp`` — ``mu = urefs[0] + MLP([single(x), e, psi(xrefs)])``.

    Same control law as CLActor, but one generic MLP over the whole (encoded)
    observation instead of the bilinear W1/W2 factorization — the deliberate
    architectural contrast. Reuses the same ``Feats`` maps and sequence encoder,
    so the two backbones differ only in how they combine the same inputs.

    Log-prob stays correct: the uref shift happens inside ``compute()``, so the
    mean GaussianMixin samples from already includes it.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = _MIN_LOG_STD,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        u_dim = self.window.u_dim
        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        self.psi = PreviewSequenceEncoder(
            preview_dim=self.window.length * self.feats.pair_dim,
            point_dim=self.feats.pair_dim, hidden=encoder_hidden, out_dim=encoder_hidden,
            mode=encoder, stride=encoder_stride)
        self.net = MLP(self.feats.single_dim + self.feats.error_dim + encoder_hidden,
                       list(hidden_dim or [128, 128]), u_dim, activation=act_module)
        self.log_std_parameter = nn.Parameter(torch.full((u_dim,), float(initial_log_std)))
        self.to(self.device)

    def _net_in(self, obs):
        n = obs.shape[0]
        x, xrefs, urefs = self.window.split(obs)
        e = self.feats.error(x, xrefs[:, 0])
        psi = self.psi(self.feats.sequence(x, xrefs).reshape(n, -1))
        return torch.cat([self.feats.single(x), e, psi], dim=-1), urefs[:, 0]

    def compute(self, inputs: dict, role: str = "policy"):
        net_in, uref = self._net_in(inputs["observations"])
        return uref + self.net(net_in), {"log_std": self.log_std_parameter}


class _TanhSquashMixin:
    """Shared tanh-squash act()/log_prob machinery for bounded-action Gaussian actors.

    One copy for every squashed backbone: this correction is easy to get subtly
    wrong, so it is not copy-pasted per backbone.

    Necessary for SAC (any off-policy method with an entropy term) on a bounded
    action space. skrl's stock ``GaussianMixin.act()`` samples an unbounded
    ``Normal(mean, std)`` and merely hard-clamps the sample afterwards —
    ``log_prob`` still comes from the unclamped Normal. SAC's ``-alpha *
    log_prob`` appears in both the Bellman target and the policy loss, so an
    unbounded log_prob (growing without limit as log_std shrinks) leaves the
    automatic entropy tuning with no fixed point. That is the divergence
    mechanism — not any single hyperparameter.

    Including classes reparameterize ``u ~ Normal(mean, std)``, squash through
    tanh, rescale to ``[low, high]``, and get the standard change-of-variables
    correction applied (Haarnoja et al. 2018, eq. 21)::

        a       = low + (high - low)/2 * (tanh(u) + 1)
        log pi(a) = log Normal(u) - sum_i log(1 - tanh(u_i)^2) - sum_i log((high-low)_i / 2)

    using the numerically stable identity (avoids log(0) as tanh(u) -> +-1)::

        log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))

    Including classes must, in this order:
      1. call ``Model.__init__`` then ``GaussianMixin.__init__`` (this mixin
         reads the ``_g_*`` attributes set there);
      2. call ``self._init_tanh_squash_bounds()`` (checks the action space is
         fully bounded and registers ``_action_low``/``_action_high``);
      3. implement ``compute(inputs, role) -> (mean, {"log_std", ["residual"]})``
         where ``mean`` locates the pre-squash Normal — the *feedback* only,
         not including uref.

    Residual (uref) is added after squashing::

        action = residual + rescale_residual(tanh(u), residual)

    preserving ``u = uref + bounded_feedback`` exactly. Adding uref to ``mean``
    (before tanh) would give ``rescale(tanh(uref + feedback))`` — a saturated
    uref that returns ``rescale(tanh(uref))``, not ``uref``, at zero feedback,
    silently destroying the reference-tracking structure.

    ``rescale_residual`` is asymmetric: ``tanh(u) >= 0`` maps into
    ``[0, high - residual]`` and ``tanh(u) < 0`` into ``[-(residual - low), 0]``,
    a different per-sample scale each side (residual varies per state). For
    every residual in [low, high] this guarantees ``tanh(u) == 0 => action ==
    residual`` exactly, and ``action`` inside ``(low, high)`` for every ``u``
    with no post-hoc clamping. A single constant rescale-then-add-then-clamp
    would push the sum out of bounds whenever residual is off-center, silently
    decoupling the clamped action from the log_prob of the unclamped one — SAC's
    entropy term would score the wrong sample. Both half-scales are positive
    constants w.r.t. ``u``, so the change-of-variables Jacobian is as simple as
    the fixed-scale case (just swap in the applicable half-scale): an exact
    closed-form log_prob for the applied action, not an approximation.

    act() is overridden entirely, not just compute(): squash-then-correct
    happens between sampling and log_prob, which overriding compute() alone
    can't express under skrl's GaussianMixin.act().

    get_entropy() is deliberately left meaningless — the squashed distribution
    has no closed-form entropy. Use only with algorithms relying on the sampled
    log_prob (SAC), never an analytic entropy bonus (PPO's entropy_loss_scale).
    """

    _RESCALE_EPS = 1e-6  # keeps half-scales/atanh args away from 0 / +-1 (log(0)/atanh divergence)

    def _init_tanh_squash_bounds(self) -> None:
        if self._g_min_actions is None or self._g_max_actions is None:
            raise ValueError(
                f"{type(self).__name__} requires a fully-bounded action space "
                "(every dimension needs a finite low/high) — tanh-squashing has "
                "nothing to rescale into otherwise."
            )
        self.register_buffer("_action_low", self._g_min_actions.clone())
        self.register_buffer("_action_high", self._g_max_actions.clone())

    def _rescale(self, tanh_u: torch.Tensor) -> torch.Tensor:
        """(-1, 1) -> [low, high]."""
        return self._action_low + 0.5 * (tanh_u + 1.0) * (self._action_high - self._action_low)

    def _unrescale(self, action: torch.Tensor) -> torch.Tensor:
        """[low, high] -> (-1, 1), clamped away from the boundary (atanh diverges there)."""
        frac = (action - self._action_low) / (self._action_high - self._action_low)
        return torch.clamp(2.0 * frac - 1.0, -1.0 + 1e-6, 1.0 - 1e-6)

    def _rescale_residual(self, tanh_u: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(-1, 1) -> (low, high) via an asymmetric, residual-centered rescale.

        Returns ``(action, scale)`` where ``scale`` is the per-sample,
        per-side half-width used (needed for the log-det correction in
        ``act()``). See the class docstring for why this is exact and
        clamp-free, unlike a fixed-scale rescale-then-add. Delegates to
        ``math_utils.rescale_residual`` — shared with ``CLActor.mean_control_squashed``.
        """
        return rescale_residual(tanh_u, residual, self._action_low, self._action_high)

    def _unrescale_residual(self, action: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of ``_rescale_residual``: action -> (tanh_u in (-1, 1), scale)."""
        feedback = action - residual
        s_hi = torch.clamp(self._action_high - residual, min=self._RESCALE_EPS)
        s_lo = torch.clamp(residual - self._action_low, min=self._RESCALE_EPS)
        scale = torch.where(feedback >= 0, s_hi, s_lo)
        tanh_u = torch.clamp(feedback / scale, -1.0 + self._RESCALE_EPS, 1.0 - self._RESCALE_EPS)
        return tanh_u, scale

    def act(self, inputs: dict, *, role: str = "") -> tuple[torch.Tensor, dict]:
        mean, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        self._g_distribution = Normal(mean, log_std.exp())

        # Optional post-squash residual (e.g. uref for the [x, xref, uref]
        # path-tracking layout). Added to the action after squashing — see the
        # class docstring for why (residual law preservation + exact,
        # clamp-free log_prob). Popped so it doesn't leak into the returned
        # outputs dict.
        residual = outputs.pop("residual", None)

        taken_actions = inputs.get("taken_actions")
        if taken_actions is not None:
            # Recompute log_prob for an already-taken (post-squash, post-rescale,
            # post-residual) action — e.g. an on-policy update replaying stored
            # actions. SAC itself never hits this path (it always samples fresh).
            actions = taken_actions
            if residual is None:
                tanh_u = self._unrescale(actions)
                scale = 0.5 * (self._action_high - self._action_low)
            else:
                tanh_u, scale = self._unrescale_residual(actions, residual)
            u = torch.atanh(tanh_u)
        else:
            u = self._g_distribution.rsample()
            tanh_u = torch.tanh(u)
            if residual is None:
                actions = self._rescale(tanh_u)
                scale = 0.5 * (self._action_high - self._action_low)
            else:
                actions, scale = self._rescale_residual(tanh_u, residual)

        log_prob = self._g_distribution.log_prob(u)
        # Change-of-variables correction for y = tanh(u), stable form of
        # log(1 - tanh(u)^2) (Haarnoja et al. 2018, eq. 21).
        log_prob = log_prob - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        # Second correction for the rescale tanh_u -> action: d(action)/d(tanh_u)
        # is `scale` per dimension — a constant half-width (0.5*(high-low)) with
        # no residual, or the residual-dependent half-width from
        # _rescale_residual/_unrescale_residual above. Either way this is exact
        # for the action actually returned below (no clamping follows).
        log_prob = log_prob - torch.log(scale)

        if self._g_reduction is not None:
            log_prob = self._g_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["log_prob"] = log_prob
        tanh_mean = torch.tanh(mean)
        mean_action = self._rescale(tanh_mean) if residual is None else self._rescale_residual(tanh_mean, residual)[0]

        outputs["mean_actions"] = mean_action
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        raise NotImplementedError(
            f"{type(self).__name__} has no closed-form entropy (the squashed "
            "distribution isn't Gaussian) — it must only be used with algorithms "
            "that rely on the sampled log_prob (SAC), not an analytic entropy "
            "bonus (e.g. PPO's entropy_loss_scale)."
        )


class SquashedMLPActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed plain-MLP actor — ``backbone: mlp-squashed``.

    ``action = urefs[0] + rescale(tanh(MLP(...) + noise))``, uref added after
    squashing (see ``_TanhSquashMixin`` — before tanh would saturate uref).

    log_std is state-Dependent (the network outputs both), the SAC convention —
    unlike the other actors' single global log_std_parameter, which is fine for
    PPO's trust-region updates but not for SAC's entropy tuning.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        hidden_dim: list | None = None,
        activation: str = "relu",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self._init_tanh_squash_bounds()

        self.window, self.feats = _layout(observation_space, x_dim, angle_idx, sym)
        act_dim = int(self.action_space.shape[0])
        act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
            if isinstance(activation, str) else activation
        self.psi = PreviewSequenceEncoder(
            preview_dim=self.window.length * self.feats.pair_dim,
            point_dim=self.feats.pair_dim, hidden=encoder_hidden, out_dim=encoder_hidden,
            mode=encoder, stride=encoder_stride)
        self.net = MLP(self.feats.single_dim + self.feats.error_dim + encoder_hidden,
                       list(hidden_dim or [256, 256]), output_dim=None, activation=act_module)
        trunk_dim = self.net.output_dim
        self.mean_head = nn.Linear(trunk_dim, act_dim)
        self.log_std_head = nn.Linear(trunk_dim, act_dim)
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.01)
        self.mean_head.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.01)
        self.log_std_head.bias.data.fill_(0.0)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        obs = inputs["observations"]
        n = obs.shape[0]
        x, xrefs, urefs = self.window.split(obs)
        e = self.feats.error(x, xrefs[:, 0])
        psi = self.psi(self.feats.sequence(x, xrefs).reshape(n, -1))
        features = self.net(torch.cat([self.feats.single(x), e, psi], dim=-1))
        mean = self.mean_head(features)  # pre-squash feedback mean (no uref)
        # urefs[0] is added after squashing (the mixin consumes "residual"), so
        # the action is uref + rescale(tanh(feedback)).
        return mean, {"log_std": self.log_std_head(features), "residual": urefs[:, 0]}


class SquashedCLActorModel(_TanhSquashMixin, GaussianMixin, Model):
    """Tanh-squashed CLActor — ``backbone: control-squashed``.

    ``action = urefs[0] + rescale(tanh(feedback + noise))`` — the feedback is
    squashed, uref added after, preserving ``u = uref + feedback`` exactly.

    ``anneal_stddev`` is always False: this backbone is for SAC, which learns
    log_std through the policy loss, and CLActor's annealing freezes ``logstd``
    and expects an external caller to step it — which would fight SAC's entropy
    optimizer. Same reason ``control-squashed`` is deliberately not in
    ``runner.CONTROL_BACKBONES`` (that set auto-enables std annealing).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        hidden_dim: list | None = None,
        activation: str = "tanh",
        x_dim: int | None = None,
        angle_idx: list | None = None,
        sym: StateSymmetry | None = None,
        encoder: str = "mlp",
        encoder_hidden: int = 64,
        encoder_stride: int = 1,
        **kwargs,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self._init_tanh_squash_bounds()
        self.cl_actor = _build_cl_actor(observation_space, action_space, x_dim, angle_idx, sym,
                                        hidden_dim, activation, False,
                                        encoder, encoder_hidden, encoder_stride)
        self.log_std_parameter = self.cl_actor.logstd
        if initial_log_std != 0.0:
            with torch.no_grad():
                self.log_std_parameter.data.fill_(initial_log_std)
        self.to(self.device)

    def compute(self, inputs: dict, role: str = "policy"):
        feedback, uref = self.cl_actor._feedback(inputs["observations"])
        return feedback, {"log_std": self.log_std_parameter, "residual": uref}


# ─────────────────────────────────────────────────────────────────────────── #
# Superseded — the flat [x, xref, uref, preview] critics. EmbeddedDeterministicModel
# inferred its layout from obs_dim; TrajectoryAwareValueModel read a separate
# privileged `states` channel and carried the "bilinear" (UVFA) and "film"
# combine modes. Both are replaced by RefWindowValueModel (concat) above.
# ─────────────────────────────────────────────────────────────────────────── #
# class EmbeddedDeterministicModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
#     """Value/critic MLP with angle-embedded input — the DeterministicMixin
#     counterpart to MLPResidualActorModel/SquashedMLPActorModel.
#
#     skrl's own model-instantiator dsl (``deterministic_model``, driven by the
#     yaml ``network: input: OBSERVATIONS`` / ``concatenate([OBSERVATIONS,
#     ACTIONS])`` keys) is vendored library code — it has no notion of an
#     angle-bearing state, so a value/critic built through it would see the raw
#     (discontinuous-at-+-pi) angle. This class is this repo's drop-in
#     replacement, wired in by CLActorRunner._component for "deterministicmixin"
#     (see runner.py) whenever the env carries a non-empty angle_idx.
#
#     Only the path-tracking observation layout ([x, xref, uref], obs_dim =
#     2*x_dim + u_dim) has a known x/xref split to embed; for any other layout
#     (e.g. velocity-tracking's flat obs) this reduces to a plain MLP over the
#     raw observation (+ actions, for a critic) — identical to what skrl's stock
#     deterministic_model would have built.
#     """
#
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         device,
#         network: list | None = None,
#         hidden_dim: list | None = None,
#         activation: str = "tanh",
#         x_dim: int | None = _X_DIM_UNSET,
#         angle_idx: list | None = None,
#         sym: StateSymmetry | None = None,
#         use_actions: bool = False,
#         analytic_potential: bool = False,
#         w_lb: float = 0.01,
#         **kwargs,
#     ):
#         Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
#         DeterministicMixin.__init__(self, clip_actions=False)
#
#         net_spec = (network or [{}])[0] if network else {}
#         hidden_dim = hidden_dim or net_spec.get("layers", [256, 256])
#         activation = net_spec.get("activations", activation)
#
#         obs_dim = int(self.observation_space.shape[0])
#         act_dim = int(self.action_space.shape[0])
#         if x_dim is _X_DIM_UNSET:
#             # Caller passed nothing (e.g. the classic CLActorRunner/yaml path) —
#             # fall back to the dimension-parity guess. Heuristic: an env whose
#             # obs_dim - act_dim happens to be even and positive for reasons
#             # unrelated to a uref layout (e.g. a flat velocity-tracking obs)
#             # would be misclassified — pass x_dim explicitly (even x_dim=None,
#             # for "no, definitely not path-tracking") to avoid this.
#             remainder = obs_dim - act_dim
#             is_path_tracking = remainder > 0 and remainder % 2 == 0
#             x_dim = remainder // 2 if is_path_tracking else None
#         else:
#             # Caller already knows the layout (e.g. contraction_runner.py's
#             # raw_env.x_dim, which is None for envs like vel-tracking that
#             # never declare it) — trust it instead of guessing from dimensions.
#             is_path_tracking = x_dim is not None
#         self._x_dim = x_dim if is_path_tracking else None
#         self._u_dim = act_dim if is_path_tracking else None
#         self._angle_idx = (angle_idx or []) if is_path_tracking else []
#         # Same gate as angle_idx: only the [x, xref, uref] layout has a
#         # position block to quotient out; a flat obs leaves pos_dim at 0.
#         self._sym = sym if is_path_tracking else None
#         self._use_actions = use_actions
#         # Future-uref preview tail (0 without preview / for a flat obs). The
#         # critic is the essential consumer: V^pi depends on the future
#         # reference, so previewing it is what makes a high discount_factor a
#         # well-posed (Markov) value target rather than a POMDP.
#         self._preview_dim = _preview_width(obs_dim, self._x_dim, act_dim) if is_path_tracking else 0
#         # O6 analytic-potential critic (see _AnalyticPotentialMixin). Needs a
#         # path-tracking [x, xref, ...] layout to recover e = x - xref at all.
#         self.use_analytic_potential = bool(analytic_potential) and is_path_tracking
#         self._init_potential(self._x_dim or 0, self._angle_idx, w_lb)
#
#         act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
#             if isinstance(activation, str) else activation
#
#         obs_in_dim = (
#             (sym.pair_dim() if sym is not None else 2 * embedded_dim(self._x_dim, self._angle_idx)) + self._u_dim + self._preview_dim
#             if is_path_tracking else obs_dim
#         )
#         net_in_dim = obs_in_dim + (act_dim if use_actions else 0)
#         self.net = MLP(net_in_dim, list(hidden_dim), 1, activation=act_module)
#
#         self.to(self.device)
#
#     def _embed_obs(self, obs: torch.Tensor) -> torch.Tensor:
#         if self._u_dim is None:
#             return obs
#         x = obs[:, : self._x_dim]
#         xref = obs[:, self._x_dim : 2 * self._x_dim]
#         uref = obs[:, 2 * self._x_dim : 2 * self._x_dim + self._u_dim]
#         preview = obs[:, 2 * self._x_dim + self._u_dim:]  # future-uref tail
#         return torch.cat(
#             [(self._sym.pair_features(x, xref) if self._sym is not None else torch.cat([embed_angles(x, self._angle_idx), embed_angles(xref, self._angle_idx)], -1)), uref, preview], dim=-1
#         )
#
#     def compute(self, inputs: dict, role: str = "value"):
#         obs_emb = self._embed_obs(inputs["observations"])
#         if self._use_actions:
#             net_in = torch.cat([obs_emb, inputs["taken_actions"]], dim=-1)
#         else:
#             net_in = obs_emb
#         value = self.net(net_in)
#         if getattr(self, "use_analytic_potential", False):
#             pot = self._analytic_potential(inputs)
#             if pot is not None:
#                 value = value + pot        # V = f_theta(s) + ||e||^2_M  (= -Phi(s))
#         return value, {}
#
#
# class TrajectoryAwareValueModel(_AnalyticPotentialMixin, DeterministicMixin, Model):
#     """Asymmetric (privileged) PPO critic: V(x, future-xref trajectory), read
#     from skrl's separate ``states``/``state_space`` channel — see
#     ``env_base.BaseEnv.configure_value_state``/``state()`` — completely
#     Decoupled from whatever preview the actor's own ``observations`` happens
#     to carry. skrl's PPO already passes both ``observations`` and ``states``
#     to every model's ``inputs`` dict (``ppo.py``'s ``act()``); this is simply
#     the first model here that reads ``inputs["states"]`` instead of
#     ``inputs["observations"]``. The actor is completely unaffected either way.
#
#     Input layout (see ``BaseEnv.state()``): ``[x (x_dim), future-xref-relative
#     points (P * x_dim)]`` — P is whatever ``configure_value_state`` was set up
#     with (a handful of geometric offsets, or every remaining step for
#     ``full_trajectory=True``), independent of the actor's own preview length.
#
#     ``encoder`` picks how the P-point trajectory block is turned into a fixed
#     vector: "mlp" (flatten), "gru", or "attn" — all three route through
#     ``PreviewSequenceEncoder`` and share ``encoder_stride`` (keep every
#     stride-th point, nearest-first; 1 = dense/every point). Stride is what
#     bounds "mlp"'s input width on a long/full-trajectory window — otherwise
#     it scales with episode length.
#
#     ``combine`` picks how the state embedding phi(x) and the goal/trajectory
#     embedding psi(traj) are turned into a scalar value:
#
#       "concat" (default) — cat([phi, psi]) through one joint MLP. Free to mix
#       the embeddings however it likes, so it can (and empirically does) fit
#       specific (x, traj) pairs jointly rather than learning independently
#       meaningful phi/psi — not actually factorized despite the separate encoders.
#
#       "bilinear" — true UVFA factorization (Schaul et al. 2015):
#       V = w^T(phi ⊙ psi) + u^T phi + v^T psi + b. Never concatenated, so the two
#       interact only through the elementwise product and each embedding must be
#       useful on its own (the additive terms cover state-only / goal-only value).
#       This is what should generalize to a new (x, traj) pair resembling two
#       different training points — concat+MLP has no structural pressure to.
#
#       "film" — middle ground, mirroring the actor's FiLMResidual: psi modulates
#       phi before a small joint head, V = head(phi * softplus(gamma(psi)) +
#       beta(psi)). Keeps a nonlinear head after the interaction (no
#       encoder_hidden rank ceiling, unlike bilinear) while still forcing the
#       interaction through an explicit trajectory-conditioned gate (unlike
#       concat). Untested at scale.
#     """
#
#     _COMBINE_MODES = ("concat", "bilinear", "film")
#
#     def __init__(
#         self,
#         state_space,
#         action_space,
#         device,
#         network: list | None = None,
#         hidden_dim: list | None = None,
#         activation: str = "tanh",
#         x_dim: int | None = None,
#         angle_idx: list | None = None,
#         encoder: str = "mlp",
#         encoder_hidden: int = 64,
#         encoder_stride: int = 1,
#         combine: str = "concat",
#         analytic_potential: bool = False,
#         w_lb: float = 0.01,
#         **kwargs,
#     ):
#         Model.__init__(self, observation_space=state_space, action_space=action_space, device=device)
#         DeterministicMixin.__init__(self, clip_actions=False)
#
#         if x_dim is None:
#             raise ValueError("TrajectoryAwareValueModel needs x_dim — the state layout is "
#                               "[x, P future-xref-relative points], not self-describing without it.")
#         self.x_dim = int(x_dim)
#         self.angle_idx = list(angle_idx or [])
#         if combine not in self._COMBINE_MODES:
#             raise ValueError(f"TrajectoryAwareValueModel: combine must be one of "
#                              f"{self._COMBINE_MODES}, got {combine!r}")
#         self.combine = combine
#
#         net_spec = (network or [{}])[0] if network else {}
#         hidden_dim = hidden_dim or net_spec.get("layers", [256, 256])
#         activation = net_spec.get("activations", activation)
#         act_module = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation.lower()] \
#             if isinstance(activation, str) else activation
#
#         state_dim = int(self.observation_space.shape[0])
#         self._traj_dim = state_dim - self.x_dim
#         if encoder not in PreviewSequenceEncoder.MODES:
#             raise ValueError(f"TrajectoryAwareValueModel: encoder must be one of "
#                              f"{PreviewSequenceEncoder.MODES}, got {encoder!r}")
#         self.traj_encoder = PreviewSequenceEncoder(
#             self._traj_dim, point_dim=self.x_dim, hidden=encoder_hidden,
#             out_dim=encoder_hidden, mode=encoder, stride=encoder_stride)
#
#         if self.combine == "concat":
#             self.net = MLP(embedded_dim(self.x_dim, self.angle_idx) + encoder_hidden,
#                            list(hidden_dim), 1, activation=act_module)
#         elif self.combine == "film":
#             # phi(x) gets the same depth as "bilinear"'s phi; psi(traj) (already
#             # produced by traj_encoder above) drives a FiLM gate on phi before a
#             # small joint head -- see class docstring's "film" entry.
#             self.phi = MLP(embedded_dim(self.x_dim, self.angle_idx),
#                            list(hidden_dim), encoder_hidden, activation=act_module)
#             self.film_gamma = nn.Linear(encoder_hidden, encoder_hidden)
#             self.film_beta = nn.Linear(encoder_hidden, encoder_hidden)
#             self.film_head = MLP(encoder_hidden, [encoder_hidden], 1, activation=act_module)
#         else:
#             # phi(x): state embedding, same width as psi(traj) (encoder_hidden)
#             # so they can be combined elementwise.
#             #
#             # Embedding width matters here, much more than it does for "concat".
#             # The value this head can represent is
#             #     V = w^T(phi ⊙ psi) + u^T phi + v^T psi + b,
#             # i.e. a diagonal bilinear form whose rank is bounded by
#             # encoder_hidden — that width is the entire capacity of the
#             # state x goal interaction, exactly the quantity UVFA (Schaul et al.
#             # 2015) treats as a primary hyperparameter. "concat" has no such
#             # bottleneck: it feeds both embeddings into a full hidden_dim MLP.
#             # So comparing the two at a shared, small encoder_hidden confounds
#             # "factorization" with "starved factorization" — the reason
#             # encoder_hidden is plumbed through to the CLI/YAML rather than left
#             # at its 64 default. phi gets hidden_dim depth too, so the two modes
#             # are matched on everything except the interaction structure.
#             self.phi = MLP(embedded_dim(self.x_dim, self.angle_idx),
#                            list(hidden_dim), encoder_hidden, activation=act_module)
#             self.bilinear_head = nn.Linear(encoder_hidden, 1)
#             self.phi_head = nn.Linear(encoder_hidden, 1)
#             self.psi_head = nn.Linear(encoder_hidden, 1)
#         self.use_analytic_potential = bool(analytic_potential)
#         self._init_potential(self.x_dim, self.angle_idx, w_lb)
#         self.to(self.device)
#
#     def compute(self, inputs: dict, role: str = "value"):
#         state = inputs["states"]
#         x = state[:, : self.x_dim]
#         traj = state[:, self.x_dim:]
#         x_emb = embed_angles(x, self.angle_idx)
#         traj_emb = self.traj_encoder(traj)
#         if self.combine == "concat":
#             value = self.net(torch.cat([x_emb, traj_emb], dim=-1))
#         elif self.combine == "film":
#             phi = self.phi(x_emb)
#             gamma = F.softplus(self.film_gamma(traj_emb))  # positive scale, mirrors actor's FiLM gain
#             beta = self.film_beta(traj_emb)
#             value = self.film_head(phi * gamma + beta)
#         else:
#             phi = self.phi(x_emb)
#             value = self.bilinear_head(phi * traj_emb) + self.phi_head(phi) + self.psi_head(traj_emb)
#         if self.use_analytic_potential:
#             pot = self._analytic_potential(inputs)
#             if pot is not None:
#                 value = value + pot        # V = f_theta(s) + ||e||^2_M  (= -Phi(s))
#         return value, {}
#
