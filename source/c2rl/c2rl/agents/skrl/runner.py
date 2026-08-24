"""CLActorRunner — backbone-aware GaussianMixin factory.

YAML policy config:

    policy:
      class: GaussianMixin
      backbone: control           # CLActor (W2 @ tanh(W1 @ (x-x_ref))); alias: contraction
      clip_log_std: True
      min_log_std: -4.605
      max_log_std: 2.0
      network:                    # layers used for w1 and w2 MLPs
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: mlp               # standard skrl MLP (default)
      network:
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: control-squashed  # tanh-squashed CLActor (SquashedCLActorModel) — same
                                   # bilinear feedback architecture as "control", but bounded
                                   # + log_prob-corrected; requires the same [x, xref, uref]
                                   # layout as "control". See models.py's SquashedCLActorModel /
                                   # _TanhSquashMixin docstrings for why this is required for
                                   # SAC (skrl's stock GaussianMixin + clip_actions is not
                                   # equivalent — clip never reaches log_prob).
      clip_log_std: True
      min_log_std: -20.0
      max_log_std: 2.0
      network:
        - name: net
          input: OBSERVATIONS
          layers: [128, 128]
          activations: tanh
      output: ACTIONS

    policy:
      class: GaussianMixin
      backbone: mlp-squashed      # tanh-squashed plain-MLP actor (SquashedMLPActorModel).
                                   # Adds uref like "mlp" does if the observation layout has one
                                   # ([x, xref, uref]); otherwise (e.g. velocity-tracking) it's a
                                   # plain squashed MLP over the full observation. State-dependent
                                   # log_std, so no initial_log_std/clip_actions (silently
                                   # ignored — squashing already bounds every sampled action).
      clip_log_std: True
      min_log_std: -20.0
      max_log_std: 2.0
      network:
        - name: net
          input: OBSERVATIONS
          layers: [256, 256]
          activations: relu
      output: ACTIONS

When backbone is ``contraction``, ``layers`` is extracted from the first
network entry and passed as ``hidden_dim`` to CLActorModel.  All other
``output`` is dropped; ``initial_log_std`` is forwarded and honoured.
"""
from __future__ import annotations

from skrl.utils.runner.torch import Runner

# Backbone → action-distribution family. Unbounded backbones sample from a raw
# Normal (valid only for PPO-family trust-region methods); squashed backbones
# tanh-bound the action + correct log_prob (valid only for SAC-family off-policy
# entropy tuning). Mixing them across families silently mistrains — see
# _assert_backbone_algo_compatible.
_UNBOUNDED_BACKBONES = frozenset({"mlp", "control", "contraction"})
_SQUASHED_BACKBONES = frozenset({"mlp-squashed", "control-squashed"})

# The CLActor backbone ("control", with "contraction" kept as a backward-
# compatible alias) always freezes its log_std parameter (CLActor's own
# anneal_stddev=True -> requires_grad=False, see nn_modules.py) since it's
# meant to be annealed on a fixed schedule rather than learned by PPO's
# gradient. Exported so callers (train.py, c2rl.py) can auto-enable
# std-dev annealing purely from the backbone choice instead of a separate
# yaml on/off flag — see agent_patches.patch_ppo_std_annealing.
CONTROL_BACKBONES = frozenset({"control", "contraction"})


def _assert_backbone_algo_compatible(backbone: str, agent_class: str | None) -> None:
    """Raise on (algorithm, backbone) pairings that would silently mistrain.

    - SAC / C2RL-SAC need a bounded (tanh-squashed) action distribution: an
      unbounded Gaussian's log_prob is unbounded below, so SAC's automatic
      entropy-coefficient tuning has no fixed point and diverges. Unbounded
      backbones (mlp / control / contraction) are rejected for SAC.
    - PPO / C2RL-PPO need a closed-form entropy (entropy bonus) and clean
      trust-region log_prob ratios; the squashed backbones deliberately raise
      from get_entropy() and have zero-gradient action boundaries, so they are
      rejected for PPO.

    Other agents (C3M, LQR, SD-LQR) are unconstrained — they don't build their
    policy through this factory with a squashed backbone and don't do SAC-style
    entropy tuning. ``agent_class=None`` also skips the check (nothing to
    validate against).
    """
    if not agent_class:
        return
    algo = agent_class.upper()
    is_sac = "SAC" in algo
    is_ppo = "PPO" in algo
    if is_sac and backbone in _UNBOUNDED_BACKBONES:
        suggested = "mlp-squashed" if backbone == "mlp" else "control-squashed"
        raise ValueError(
            f"backbone: {backbone!r} is an UNBOUNDED Gaussian and is not valid for "
            f"agent class {agent_class!r} (SAC family): SAC's automatic entropy tuning "
            f"has no fixed point with an unbounded log_prob and will diverge. Use "
            f"backbone: {suggested!r} (tanh-squashed, bounded) for SAC."
        )
    if is_ppo and backbone in _SQUASHED_BACKBONES:
        suggested = "mlp" if backbone == "mlp-squashed" else "control"
        raise ValueError(
            f"backbone: {backbone!r} is tanh-squashed and has no closed-form entropy "
            f"(get_entropy raises), so it is not valid for agent class {agent_class!r} "
            f"(PPO family), whose entropy bonus / analytic entropy require it. Use "
            f"backbone: {suggested!r} (unbounded Gaussian) for PPO."
        )


def _is_window_space(space) -> bool:
    """True for a ``{x, xrefs, urefs}`` path-tracking observation. Velocity-
    tracking envs keep a flat Box obs and route to skrl's stock models."""
    import gymnasium as gym
    return isinstance(space, gym.spaces.Dict) and {"x", "xrefs", "urefs"} <= set(space.spaces)


def _gaussian_factory(observation_space, state_space, action_space, device,
                       backbone: str = "mlp", agent_class: str | None = None, **kwargs):
    # Hard guard: reject algorithm/backbone pairs that silently mistrain
    # (unbounded + SAC → divergence; squashed + PPO → no analytic entropy).
    _assert_backbone_algo_compatible(backbone, agent_class)

    # x_dim is now only a cross-Check against what the observation space
    # declares (models raise on a mismatch) — never the source of the layout.
    x_dim = kwargs.pop("x_dim", None)
    is_window = _is_window_space(observation_space)

    def _hidden(default):
        network = kwargs.pop("network", [{}])
        hd = network[0].get("layers", default) if network else default
        act = network[0].get("activations", None) if network else None
        kwargs.pop("output", None)
        return hd, act

    # "control" is the preferred spelling for the CLActor backbone;
    # "contraction" is kept as a backward-compatible alias.
    if backbone in ("control", "contraction", "control-squashed"):
        if not is_window:
            raise ValueError(
                f"backbone: {backbone} requires a {{x, xrefs, urefs}} observation "
                f"(a path-tracking env), got {type(observation_space).__name__}. "
                f"Use backbone: mlp / mlp-squashed for velocity-tracking envs.")
        from c2rl.agents.skrl.models import CLActorModel, SquashedCLActorModel
        hidden_dim, _ = _hidden([128, 128])
        if backbone == "control-squashed":
            return SquashedCLActorModel(
                observation_space=observation_space, action_space=action_space,
                device=device, hidden_dim=hidden_dim, x_dim=x_dim, **kwargs)
        # initial_log_std is forwarded, not dropped: CLActorModel declares the
        # argument and applies it. Popping it here pinned every backbone: control
        # policy to log_std 0.0 (sigma = 1.0) regardless of the configured value.
        return CLActorModel(
            observation_space=observation_space, action_space=action_space,
            device=device, hidden_dim=hidden_dim, x_dim=x_dim, **kwargs)

    if backbone == "mlp-squashed" and is_window:
        from c2rl.agents.skrl.models import SquashedMLPActorModel
        hidden_dim, activation = _hidden([256, 256])
        # log_std is state-dependent (network head) for this backbone, and
        # squashing already bounds every sampled action — these two yaml keys
        # (meaningful for the other backbones) don't apply here.
        kwargs.pop("initial_log_std", None)
        kwargs.pop("clip_actions", None)
        return SquashedMLPActorModel(
            observation_space=observation_space, action_space=action_space, device=device,
            hidden_dim=hidden_dim, activation=activation or "relu", x_dim=x_dim, **kwargs)

    if backbone == "mlp" and is_window:
        # "mlp" still adds urefs[0] to its output (mu = uref + MLP(...)), just via
        # a generic MLP over the encoded observation instead of CLActor's bilinear
        # W1/W2 structure. Vel-tracking envs (flat Box obs) fall through below.
        from c2rl.agents.skrl.models import MLPResidualActorModel
        hidden_dim, _ = _hidden([128, 128])
        initial_log_std = kwargs.pop("initial_log_std", 0.0)
        return MLPResidualActorModel(
            observation_space=observation_space, action_space=action_space, device=device,
            hidden_dim=hidden_dim, initial_log_std=initial_log_std, x_dim=x_dim, **kwargs)

    from skrl.utils.model_instantiators.torch import gaussian_model
    for _k in ("angle_idx", "sym", "encoder", "encoder_hidden", "encoder_stride"):
        kwargs.pop(_k, None)
    return gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        **kwargs,
    )


def _deterministic_factory(observation_space, state_space, action_space, device, **kwargs):
    """DeterministicMixin (value/critic) factory — the RefWindowValueModel
    counterpart to ``_gaussian_factory`` above.

    On a ``{x, xrefs, urefs}`` observation this builds the structured critic
    ``V = MLP([phi(x, e) || psi(xrefs)])`` — the mirror of the actor's
    ``W1(x)`` / ``W2(xrefs)`` split, with the reference window going through the
    same ``--encoder`` (mlp | gru | attn) and the same relative-position /
    wrapped-angle feature map. Velocity-tracking envs (flat Box obs) fall
    through to skrl's stock ``deterministic_model``, which is what they used
    before — it has no notion of an angle-bearing state, but they have none.
    """
    network = kwargs.pop("network", [{}])
    kwargs.pop("output", None)
    net_spec = network[0] if network else {}
    use_actions = "ACTIONS" in str(net_spec.get("input", "OBSERVATIONS"))
    hidden_dim = net_spec.get("layers", [256, 256])
    activation = net_spec.get("activations", "tanh")

    if not _is_window_space(observation_space):
        from skrl.utils.model_instantiators.torch import deterministic_model
        for _k in ("angle_idx", "sym", "x_dim", "encoder", "encoder_hidden",
                   "encoder_stride", "combine", "analytic_potential", "w_lb"):
            kwargs.pop(_k, None)
        return deterministic_model(
            observation_space=observation_space, state_space=state_space,
            action_space=action_space, device=device,
            network=network or [{}], output="ONE", **kwargs)

    from c2rl.agents.skrl.models import RefWindowValueModel

    # yaml spells psi's width "embed_dim" (matching --critic_embed_dim); the
    # model calls it encoder_hidden, shared with the actor's encoder.
    if "embed_dim" in kwargs:
        kwargs.setdefault("encoder_hidden", int(kwargs.pop("embed_dim")))

    return RefWindowValueModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        hidden_dim=hidden_dim,
        activation=activation,
        use_actions=use_actions,
        **kwargs,
    )


class CLActorRunner(Runner):
    def __init__(self, env, cfg):
        # Capture the agent class up front so the gaussian factory can reject
        # (algorithm, backbone) combinations that silently mistrain (see
        # _assert_backbone_algo_compatible). Set before super().__init__ because
        # that's what triggers model construction via _component below.
        self._agent_class = str((cfg.get("agent") or {}).get("class", "")).strip()
        super().__init__(env, cfg)

    def _component(self, name: str):
        if name.lower() == "gaussianmixin":
            agent_class = getattr(self, "_agent_class", "")

            def _factory(*args, **kw):
                return _gaussian_factory(*args, agent_class=agent_class, **kw)

            return _factory
        if name.lower() == "deterministicmixin":
            return _deterministic_factory
        return super()._component(name)
