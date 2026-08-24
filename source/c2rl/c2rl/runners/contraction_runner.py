"""ContractionRunner — single unified Runner for all algorithms × all envs.

Mirrors the skrl Runner interface::

    runner = ContractionRunner(env, cfg, task_id="Car-Direct-v0",
                               num_envs=4, is_classic=True)
    runner.run()

Algorithm routing
-----------------
PPO / SAC (and any other skrl-native algo):
    → skrl Runner  (skrl.utils.runner.torch.Runner)

C3M / LQR / SDLQR / C2RL-PPO / C2RL-SAC:
    → native skrl Agent subclasses (C3MAgent, LQRAgent, SDLQRAgent, C2RLAgent)
    with custom skrl Trainers (C3MSkrlTrainer, C2RLSkrlTrainer, or
    SequentialTrainer for eval-only analytical agents). C2RL-PPO/C2RL-SAC are
    the same C2RLAgent class — the algo string just selects which
    base_algorithm (PPO or SAC) its single deployed policy is built on. C2RL
    always synthesizes an offline, frozen CMG network (Neural Contraction
    Metric); ``cmg_method`` selects how it's trained ("ccm" C1/C2 loss
    minimization, or "cvstem" SDP regression). See _setup_c2rl.

Classic env: internally creates SyncVectorEnv + wrap_env.
Isaac env:   env is already a SkrlVecEnvWrapper — passed through.

Custom YAML flags handled here (stripped before passing to skrl)
-----------------------------------------------------------------
use_state_norm: true/false  →  observation_preprocessor
use_value_norm: true/false  →  value_preprocessor (PPO only)
anneal_log_std: true/false  →  exponential log_std schedule 0 → -20
"""

from __future__ import annotations

import copy
import os

import gymnasium as gym

from c2rl.agents.skrl.rl_glue import filter_cfg_fields as _filter_cfg_fields

_SKRL_ALGOS = frozenset({"ppo", "sac", "td3", "ddpg", "amp", "ippo", "mappo"})
_CONTRACTION_ALGOS = frozenset({"c3m", "lqr", "sdlqr", "cvstem-lqr", "c2rl-ppo", "c2rl-sac"})

# Contraction algorithms that build and train no dynamics network of their own:
# they read f/B straight off ``get_f_and_B``. Classic envs answer that
# analytically; Isaac envs have no closed form, so these need a pretrained
# NeuralDynamics injected into the env (see _resolve_dynamics_model).
_ANALYTIC_CONTROLLERS = frozenset({"lqr", "sdlqr", "cvstem-lqr"})


def _unwrap_env(env):
    """Peel every wrapper layer down to the concrete env instance.

    Follows ``.unwrapped`` then ``.env``, guarding against wrappers that return
    themselves (and against cycles). Classic envs bottom out at the gymnasium
    env; Isaac envs at the DirectRLEnv subclass under SkrlVecEnvWrapper.
    """
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        inner = getattr(env, "unwrapped", None) or getattr(env, "env", None)
        if inner is None or inner is env:
            break
        env = inner
    return env


def _env_attrs(raw_env, *names, default=None):
    """Read attributes off ``raw_env``, falling back to its first sub-env.

    A gymnasium ``SyncVectorEnv`` holds N independent Python envs in ``.envs``
    and forwards none of their attributes, so the values contraction agents need
    (``x_dim``/``u_dim``/``angle_idx``/``get_f_and_B``/``get_rollout``) live one
    level deeper there than they do on a batched Isaac env.
    """
    sub = next(iter(getattr(raw_env, "envs", []) or []), None)
    sub = getattr(sub, "unwrapped", sub) if sub is not None else None
    out = []
    for name in names:
        value = getattr(raw_env, name, None)
        if value is None and sub is not None:
            value = getattr(sub, name, None)
        out.append(default if value is None else value)
    return tuple(out)


def _make_cmg_bounds_fn(cmg_model, w_lb: float):
    """Build the (x_batch) -> (m_bar, m_underbar) closure a path-tracking env's
    PathTracking figure uses to draw its theoretical exponential bound (see
    PathTrackingBase.set_contraction_certificate) — the current contraction
    metric's eigenvalue extremes over a batch of states, not the static
    w_lb/w_ub config bounds.
    """
    from c2rl.agents.skrl.math_utils import bound_W, spd_inverse

    ccm_gen = cmg_model.ccm_gen
    x_dim = cmg_model.x_dim

    def _bounds_fn(x_batch):
        import torch
        with torch.no_grad():
            raw_W, _ = ccm_gen(x_batch)
            # Honor the CMG's `bounded` flag so a BoundedCCM_Generator (already
            # eigenvalue-bounded) isn't double-bounded by an extra +w_lb·I —
            # otherwise the plotted contraction envelope would use a different
            # metric than the deployed one. Mirrors c2rl.py's bound_W calls.
            W = bound_W(raw_W, w_lb, x_dim, getattr(ccm_gen, "bounded", False))
            M = spd_inverse(W)
            eigvals = torch.linalg.eigvalsh(M)
        return float(eigvals.max().item()), float(eigvals.min().clamp(min=1e-12).item())

    return _bounds_fn


def _resolve_symmetry(raw_env, x_dim, angle_idx):
    """Which symmetry of the tracking problem the networks may quotient out.

    Two stages, and both must pass:

    1. The env's ``state_names`` say what each state dim is (see
       agents/skrl/state_symmetry.py). Names alone decide the candidate: a
       pos_x/pos_y pair plus a yaw and otherwise only recognised yaw-invariant
       dims -> SE(2); any pos_* dim -> translation; nothing -> none. An
       unrecognised name fails closed to translation rather than being assumed
       invariant.
    2. The candidate is checked against the env's own dynamics -- translation
       needs f, B invariant; SE(2) needs them equivariant (f(R.x) = R f(x)).
       A declaration is not a proof, and this is what catches e.g. the
       quadrotor, whose roll/pitch are world referenced so a yaw rotation mixes
       them: it is silently demoted to translation instead of mis-modelled.

    Envs with no analytical ``get_f_and_B`` (Isaac Sim) cannot be checked, so
    they keep the name-derived candidate but are capped at translation -- the
    conservative choice, since an unverifiable SE(2) claim would corrupt every
    network input if wrong.

    Returns a StateSymmetry (mode "none" disables everything) or None.
    """
    from c2rl.agents.skrl.state_symmetry import StateSymmetry

    (names,) = _env_attrs(raw_env, "state_names", default=())
    names = tuple(names or ())
    if not names or (x_dim and len(names) != int(x_dim)):
        if names:
            print(f"[ContractionRunner] state_names has {len(names)} entries but "
                  f"x_dim={x_dim} — symmetry quotient DISABLED.")
        return None

    sym = StateSymmetry.from_names(names)
    if sym.mode == "none":
        return None

    get_f_and_B, x_min, x_max = _env_attrs(raw_env, "get_f_and_B", "X_MIN", "X_MAX")
    if get_f_and_B is None or x_min is None or x_max is None:
        # Isaac Sim: no closed-form dynamics to test against, so the schema is
        # the only evidence. from_names() already refused SE(2) unless every dim
        # is a recognised yaw-invariant name, which for the quadruped/humanoid is
        # a real argument (projected_gravity_b, root_lin_vel_b, root_ang_vel_b and
        # the joint states are body frame, hence yaw invariant by construction).
        # Accepted, but announced as unverified so a wrong frame suffix on some
        # future env is discoverable from the log rather than silent.
        print(f"[ContractionRunner] symmetry quotient: {sym.describe()}  "
              f"[UNVERIFIED — no analytical f/B; trusting state_names. A "
              f"world-frame dim misnamed as body-frame (_b) would break this.]")
        return sym

    if not sym.verify_translation(get_f_and_B, x_min, x_max):
        print("[ContractionRunner] f/B DO depend on the declared position dims — "
              "symmetry quotient DISABLED (check state_names).")
        return None
    if sym.mode == "se2" and not sym.verify_rotation(get_f_and_B, x_min, x_max):
        sym = sym.demote_to_translation()
        print("[ContractionRunner] f/B are translation invariant but NOT yaw "
              "equivariant — demoted to 'translation'.")
    print(f"[ContractionRunner] symmetry quotient: {sym.describe()}")
    return sym


def _cmg_symmetry(sym):
    """The symmetry the CMG's W(x) may use — never SE(2).

    A metric is a tensor, not a scalar: under a symmetry g it must transform as
    W(g.x) = dg W(x) dg^T, so W cannot simply be fed yaw-free features the way an
    invariant control output can. Making W yaw-equivariant needs an explicit
    conjugation (build W in the canonical frame, rotate it back), which also
    changes C3M's Wdot / LMI terms — a separate change. Until then W keeps the
    translation quotient, which is a genuine invariance for it (f and B do not
    depend on position at all, so neither should the metric).
    """
    return None if sym is None else sym.demote_to_translation()


def _build_c3m_models(models_cfg: dict, agent_cfg: dict, obs_space, act_space, device,
                       x_dim, u_dim, angle_idx: list, sym=None,
                       policy_key: str = "policy") -> dict:
    """Build the policy/cmg/(optional) dynamics models for a C3M-style offline
    synthesis phase — used by ``_setup_c3m`` (pure C3M, ``policy_key="policy"``).
    Mutates ``agent_cfg`` (pops ``actor_architecture`` so it doesn't trip the
    C3MCfg field filter). The returned dict uses the keys ``"policy"``/``"cmg"``/
    (optional)``"dynamics"`` — what C3MAgent itself expects.
    """
    angle_idx = list(angle_idx or [])
    hd_policy = models_cfg.get(policy_key, {}).get("network", [{}])[0].get("layers", [128, 128])
    hd_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("layers", [128, 128])
    act_policy = models_cfg.get(policy_key, {}).get("network", [{}])[0].get("activations", "tanh")
    act_cmg = models_cfg.get("cmg", {}).get("network", [{}])[0].get("activations", "tanh")

    # agent.actor_architecture: sweep-friendly override of the *policy* network
    # spec (the CLActor's w1/w2 MLPs share these hidden layers; only their
    # input/output sizes differ). Accepts a bare hidden-layer list
    # (e.g. [256, 256]) or a dict {layers: [...], activations: "tanh"}.
    # Only the policy net is affected — not the CMG.
    actor_arch = agent_cfg.pop("actor_architecture", None)
    if isinstance(actor_arch, dict):
        hd_policy = actor_arch.get("layers", hd_policy)
        act_policy = actor_arch.get("activations", act_policy)
    elif actor_arch is not None:
        hd_policy = list(actor_arch)

    # C3M is a certificate-based controller synthesis (not a stochastic-policy
    # RL method), so the CLActor should be deterministic by default — no
    # Gaussian log_std / sampling machinery. A config may still opt into the
    # GaussianMixin wrapper explicitly.
    #
    # backbone: control-squashed selects the tanh-squashed (bounded) variant —
    # same "backbone name picks the class" convention PPO/SAC's
    # _gaussian_factory uses (see runner.py). C3M certifies whichever control
    # law the selected class's compute() returns (see _compute_loss below), so
    # squashed-vs-unbounded is entirely a function of this choice, not a
    # separate flag.
    policy_class_str = models_cfg.get(policy_key, {}).get("class", "DeterministicMixin")
    policy_backbone = models_cfg.get(policy_key, {}).get("backbone", "control")
    squashed = policy_backbone == "control-squashed"
    if policy_class_str == "GaussianMixin":
        if squashed:
            # SquashedCLActorModel.compute() deliberately returns the pre-squash
            # feedback (mean of the Normal) — squashing happens inside act(),
            # for SAC's log_prob correction (see _TanhSquashMixin). C3M
            # certifies compute()'s output directly (no sampling), so this
            # combination would silently certify an unbounded, uref-less
            # value instead of the actual bounded control law. Use the
            # DeterministicMixin default instead (policy.class unset, or
            # explicitly "DeterministicMixin") — SquashedCLDeterministicActorModel
            # squashes inside compute() itself.
            raise ValueError(
                "C3M backbone: control-squashed requires class: DeterministicMixin "
                "(the default) — GaussianMixin + control-squashed isn't supported for "
                "C3M's certification, which calls compute() directly rather than act()."
            )
        from c2rl.agents.skrl.models import CLActorModel
        policy_cls = CLActorModel
    else:
        from c2rl.agents.skrl.models import CLDeterministicActorModel, SquashedCLDeterministicActorModel
        policy_cls = SquashedCLDeterministicActorModel if squashed else CLDeterministicActorModel

    policy_kwargs = models_cfg.get(policy_key, {}).copy()
    policy_kwargs.pop("class", None)
    policy_kwargs.pop("network", None)
    policy_kwargs.pop("backbone", None)
    policy_kwargs.pop("angle_idx", None)  # angle_idx below is the single source of truth
    policy_kwargs.pop("sym", None)    # ditto: the env is the source of truth

    constrain_eigenvalues = models_cfg.get("cmg", {}).get("network", [{}])[0].get("constrain_eigenvalues", False)
    w_lb = agent_cfg.get("w_lb", 0.1)
    w_ub = agent_cfg.get("w_ub", 10.0)

    from c2rl.agents.skrl.models import MetricModel
    models = {
        "policy": policy_cls(obs_space, act_space, device, hidden_dim=hd_policy, activation=act_policy, x_dim=x_dim, angle_idx=angle_idx, sym=sym, **policy_kwargs),
        "cmg": MetricModel(obs_space, act_space, device, hidden_dim=hd_cmg, activation=act_cmg, x_dim=x_dim, angle_idx=angle_idx, sym=_cmg_symmetry(sym), constrain_eigenvalues=constrain_eigenvalues, w_lb=w_lb, w_ub=w_ub),
    }

    # Only build a NeuralDynamics when learning dynamics (use_empirical_dynamics
    # =True). Under analytical dynamics the agent ignores it (C3MAgent uses the
    # env's get_f_and_B), so building it would just waste params/memory.
    if "dynamics" in models_cfg and agent_cfg.get("use_empirical_dynamics", False):
        from c2rl.agents.skrl.nn_modules import NeuralDynamics
        dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
        dyn_hd = dyn_net.get("layers", [256, 256])
        dyn_act = dyn_net.get("activations", "relu")
        models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device, angle_idx=angle_idx, sym=_cmg_symmetry(sym))

    return models


def _build_gaussian_policy(models_cfg: dict, block_name: str, obs_space, state_space, act_space,
                           device, x_dim, angle_idx: list, agent_class: str, sym=None):
    """Build a deployed policy — same backbone dispatch PPO/SAC path-tracking
    configs use (control -> CLActorModel, mlp -> MLPResidualActorModel, both
    u = uref + feedback). Used by ``_setup_c2rl``. ``agent_class`` lets
    ``_gaussian_factory`` reject unbounded backbones for SAC-based agents
    (divergence) and squashed backbones for PPO-based ones.
    """
    from c2rl.agents.skrl.runner import _gaussian_factory

    spec = models_cfg.get(block_name, {}).copy()
    spec.pop("class", None)
    backbone = spec.pop("backbone", "control")
    spec.setdefault("clip_log_std", True)
    spec.setdefault("min_log_std", -4.605)
    spec.setdefault("max_log_std", 2.0)
    spec.setdefault("angle_idx", angle_idx)
    spec.setdefault("sym", sym)
    # x_dim is already known from the env — pass it explicitly so backbones
    # like mlp-squashed, which see both path-tracking and vel-tracking
    # layouts, don't have to guess the [x, xref, uref] split from obs_dim/
    # act_dim parity alone.
    spec.setdefault("x_dim", x_dim)
    return _gaussian_factory(obs_space, state_space, act_space, device,
                             backbone=backbone, agent_class=agent_class, **spec)


def _build_critics(models_cfg: dict, block_name: str, base_algorithm: str, obs_space, act_space,
                   device, x_dim, angle_idx: list, key_prefix: str = "", sym=None,
                   state_space=None, critic_encoder: str | None = None,
                   critic_combine: str = "concat", critic_embed_dim: int = 64,
                   critic_encoder_stride: int = 1,
                   critic_analytic_potential: bool = False, w_lb: float = 0.01) -> dict:
    """Build the value/critic model(s) for one policy — a single V(obs) model
    for PPO, or the twin-Q + target architecture (critic_1/2 + targets) for
    SAC. Used by ``_setup_c2rl`` (``key_prefix=""``, a single deployed policy).

    The critic reads the same ``{x, xrefs, urefs}`` observation as the actor
    (the privileged ``states`` channel is gone) and gets its independence from
    its own architecture: ``V = MLP([phi(x, e) || psi(xrefs)])``, mirroring the
    actor's ``W1(x)`` / ``W2(xrefs)`` split. ``critic_encoder`` selects psi's
    encoder (mlp | gru | attn); it defaults to the actor's ``--encoder``.
    """
    from c2rl.agents.skrl.models import RefWindowValueModel

    net = models_cfg.get(block_name, {}).get("network", [{}])
    hd = net[0].get("layers", [256, 256]) if net else [256, 256]
    act = net[0].get("activations", "tanh") if net else "tanh"
    common = dict(hidden_dim=hd, activation=act, x_dim=x_dim, angle_idx=angle_idx, sym=sym,
                  encoder=critic_encoder or "mlp", encoder_hidden=critic_embed_dim,
                  encoder_stride=critic_encoder_stride, combine=critic_combine)

    if base_algorithm == "PPO":
        return {f"{key_prefix}value": RefWindowValueModel(
            obs_space, act_space, device, use_actions=False,
            analytic_potential=critic_analytic_potential, w_lb=w_lb, **common,
        )}
    elif base_algorithm == "SAC":
        return {
            f"{key_prefix}{k}": RefWindowValueModel(
                obs_space, act_space, device, use_actions=True, **common)
            for k in ("critic_1", "critic_2", "target_critic_1", "target_critic_2")
        }
    else:
        raise ValueError(f"Unsupported base_algorithm: {base_algorithm}")


# ────────────────────────────────────────────────────────────────────────── #
# Helper: translate custom YAML flags to skrl format
# ────────────────────────────────────────────────────────────────────────── #

def _resolve_dynamics_model(algo: str, agent_cfg: dict, is_classic: bool, device,
                            explicit_model=None):
    """Return the ``NeuralDynamics`` an analytic controller needs, or ``None``.

    ``lqr``/``sdlqr``/``cvstem-lqr`` never train a dynamics model — they consume
    ``get_f_and_B`` directly. On a classic env that is the exact analytical
    dynamics and nothing is needed here. On an Isaac env there is no closed
    form, so a NeuralDynamics pretrained by an earlier C3M/C2RL run
    (``checkpoints/dynamics.pt``) must be loaded and injected into the env;
    otherwise ``PathTrackingBase.get_f_and_B`` raises deep inside the rollout.
    Raises up front, with the actual flag to pass, rather than letting that
    happen.
    """
    # Popped, not read: it is a runner-level knob, not a field on any agent's
    # Cfg dataclass, so leaving it in would trip filter_cfg_fields' warning.
    path = str(agent_cfg.pop("dynamics_checkpoint", "") or "")
    if explicit_model is not None:
        return explicit_model
    if path:
        from c2rl.agents.skrl.nn_modules import NeuralDynamics
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"dynamics_checkpoint '{path}' not found. It should point at a "
                f"dynamics.pt written by a previous C3M/C2RL run "
                f"(<log_dir>/checkpoints/dynamics.pt)."
            )
        return NeuralDynamics.load(path, device=str(device))
    if not is_classic and algo in _ANALYTIC_CONTROLLERS:
        raise ValueError(
            f"'{algo}' has no dynamics model of its own and Isaac Sim envs expose no "
            f"analytical get_f_and_B. Pass --dynamics_checkpoint <path-to-dynamics.pt> "
            f"(written by a previous C3M or C2RL run as <log_dir>/checkpoints/dynamics.pt), "
            f"or set agent.dynamics_checkpoint in the yaml."
        )
    return None


def _prepare_skrl_cfg(cfg: dict, algo: str) -> tuple[dict, bool]:
    """Pop C2RL-specific keys and translate to skrl conventions."""
    cfg = copy.deepcopy(cfg)
    a = cfg.setdefault("agent", {})

    a.pop("use_state_norm", None)  # state norm disabled everywhere (see module docstring)
    use_value = a.pop("use_value_norm", True)
    anneal = a.pop("anneal_log_std", False)

    for k in ("state_preprocessor", "state_preprocessor_kwargs",
              "observation_preprocessor", "observation_preprocessor_kwargs"):
        a.pop(k, None)

    if use_value and algo == "ppo":
        a["value_preprocessor"] = "RunningStandardScaler"
        a["value_preprocessor_kwargs"] = None
    else:
        a.pop("value_preprocessor", None)
        a.pop("value_preprocessor_kwargs", None)

    return cfg, anneal


# ────────────────────────────────────────────────────────────────────────── #
# Helper: PPO log_std annealing hook
# ────────────────────────────────────────────────────────────────────────── #

def _patch_ppo_annealing(runner) -> None:
    """Exponential schedule: log_std = 0*(1-t^5) + (-20)*t^5, t=progress."""
    import torch
    _orig = runner.agent.post_interaction

    def _hook(timestep: int, timesteps: int) -> None:
        _orig(timestep=timestep, timesteps=timesteps)
        ratio = (timestep / max(1, timesteps)) ** 5.0
        val = max(-20.0, min(2.0, float(-20.0 * ratio)))
        with torch.no_grad():
            runner.agent.policy.log_std_parameter.fill_(val)

    runner.agent.post_interaction = _hook


# ────────────────────────────────────────────────────────────────────────── #
# Helper: build skrl env wrapper for contraction agents
# ────────────────────────────────────────────────────────────────────────── #

def _make_skrl_env(env, task_id: str | None, num_envs: int, is_classic: bool):
    """Return a skrl-wrapped env suitable for contraction agents."""
    if is_classic:
        # Reuse the caller's env if it's already a real, usable one instead of
        # unconditionally rebuilding a bare duplicate. train.py's classic path
        # already constructs SyncVectorEnv + wrap_env + WandbPlotWrapper before
        # calling ContractionRunner — rebuilding from scratch here silently
        # discarded that wrapping, so C3M/LQR/SDLQR/C2RL's actual training loop
        # (which steps whatever this function returns, not the caller's `env`)
        # ran against an unwrapped env: no Stability/* forwarding, no
        # normalized_error/path_tracking plots. Building fresh from `task_id`
        # remains the fallback for callers that pass a bare/unvectorized env
        # (see the module docstring's documented usage).
        from ..agents.skrl.contraction_metrics import StatManagerEnvWrapper
        if hasattr(env, "step") and hasattr(env, "observation_space") and hasattr(env, "num_envs"):
            # Auto-collect path-tracking metrics on every reset/step so eval loops
            # don't hand-thread stats.update(...) (see StatManagerEnvWrapper).
            return StatManagerEnvWrapper(env) if not hasattr(env, "stability_summary") else env
        if task_id is None and hasattr(env, "spec") and env.spec:
            task_id = env.spec.id
        if task_id is None:
            raise ValueError("task_id is required for classic envs")
        from gymnasium.vector import SyncVectorEnv
        from skrl.envs.wrappers.torch import wrap_env
        vec_env = SyncVectorEnv([lambda: gym.make(task_id)] * num_envs)
        # Do not force vec_env.device = "cpu" here. The underlying physics step
        # is numpy/CPU-bound either way, but GymnasiumWrapper.step/reset already
        # bridges numpy <-> torch at the right device via tensorize_space /
        # untensorize_space (actions are .cpu().numpy()'d before being handed to
        # the numpy env; observations are re-tensorized onto `device` after).
        # Leaving vec_env.device unset lets GymnasiumWrapper fall back to the
        # global skrl device (cuda:0 when available) — matching Isaac envs and
        # letting C3M/C2RL's batched Jacobian/backprop math run on GPU instead
        # of silently pinning the whole classic pipeline (including the neural
        # networks) to CPU.
        return StatManagerEnvWrapper(wrap_env(vec_env, wrapper="gymnasium")) if not hasattr(env, "stability_summary") else wrap_env(vec_env, wrapper="gymnasium")
    return env  # Isaac: already wrapped


# ────────────────────────────────────────────────────────────────────────── #
# ContractionRunner
# ────────────────────────────────────────────────────────────────────────── #

class ContractionRunner:
    """Unified runner — same interface as ``skrl.utils.runner.torch.Runner``.

    Args:
        env: gymnasium environment (raw single env or pre-wrapped).
        cfg: agent + trainer config dict (loaded from YAML).
        task_id: gym env id; needed for SyncVectorEnv on classic envs.
        num_envs: number of parallel envs (classic envs only).
        is_classic: True for classic (no-Isaac) envs.
        ml_framework: ``"torch"`` or ``"jax"`` (skrl-native algos only).
        dynamics_model: optional NeuralDynamics for Isaac envs. Injected into
            the env via ``set_dynamics_model()`` so ``get_f_and_B`` works for
            C3M / LQR / SD-LQR / C2RL. Classic envs have analytical dynamics
            and do not need this.
    """

    def __init__(
        self,
        env,
        cfg: dict,
        *,
        task_id: str | None = None,
        num_envs: int = 1,
        is_classic: bool = True,
        ml_framework: str = "torch",
        dynamics_model=None,
    ) -> None:
        algo = cfg.get("agent", {}).get("class", "ppo").lower()

        if algo in _SKRL_ALGOS:
            self._setup_skrl(env, cfg, algo, task_id, num_envs, is_classic, ml_framework)
        elif algo in _CONTRACTION_ALGOS:
            self._setup_contraction(env, cfg, algo, task_id, num_envs, is_classic, dynamics_model)
        else:
            raise ValueError(
                f"Unknown algorithm '{algo}'. "
                f"skrl: {sorted(_SKRL_ALGOS)}, contraction: {sorted(_CONTRACTION_ALGOS)}"
            )

    # ── skrl-native algos (PPO/SAC/etc.) ───────────────────────────────── #

    def _setup_skrl(self, env, cfg, algo, task_id, num_envs, is_classic, ml_framework):
        skrl_cfg, anneal = _prepare_skrl_cfg(cfg, algo)
        skrl_cfg["trainer"]["close_environment_at_exit"] = False

        # Unwrap to the deepest raw env to check for angle_idx
        (angle_idx,) = _env_attrs(_unwrap_env(env), "angle_idx", default=[])
        angle_idx = list(angle_idx or [])

        if skrl_cfg.get("agent", {}).get("observation_preprocessor") == "RunningStandardScaler" and angle_idx:
            raise ValueError(
                "use_state_norm=True (RunningStandardScaler) is fundamentally incompatible "
                "with environments that expose angle_idx. Normalizing raw angles linearly "
                "destroys the periodicity required by embed_angles(cos, sin). "
                "Disable use_state_norm or use a custom scaler."
            )

        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)

        if ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        else:
            from skrl.utils.runner.jax import Runner

        self._runner = Runner(skrl_env, skrl_cfg)
        self._env = skrl_env
        self._mode = "skrl"

        if anneal and algo == "ppo":
            _patch_ppo_annealing(self._runner)

    # ── contraction agents (C3M/LQR/SDLQR/C2RL) ───────────────────────── #

    def _setup_contraction(self, env, cfg, algo, task_id, num_envs, is_classic, dynamics_model=None):
        skrl_env = _make_skrl_env(env, task_id, num_envs, is_classic)
        # Use the wrapped env's device for both classic and Isaac envs. The
        # classic env's own physics step is numpy/CPU-bound regardless (see
        # _make_skrl_env), but the agent's neural networks and gradient math
        # (C3M/C2RL Jacobians, batched contraction losses) benefit hugely from
        # GPU — hardcoding CPU here was pinning that compute off the GPU even
        # when one was available. SD-LQR/LQR are unaffected: they pin their own
        # internal `_compute_device = "cpu"` regardless of this value, since
        # scipy's care solver is CPU-only anyway.
        device = skrl_env.device
        obs_space = skrl_env.observation_space
        act_space = skrl_env.action_space
        state_space = getattr(skrl_env, "state_space", None)

        # Deepest raw env — the one exposing get_rollout / get_f_and_B / x_dim.
        raw_env = _unwrap_env(env)

        agent_cfg = copy.deepcopy(cfg.get("agent", {}))
        agent_cfg.pop("class", None)

        # Isaac envs have no closed-form dynamics: lqr/sdlqr/cvstem-lqr (which
        # train no dynamics model of their own) need a pretrained NeuralDynamics
        # injected here, or get_f_and_B raises mid-rollout. No-op for classic
        # envs and for C3M/C2RL, which own their dynamics model directly.
        dynamics_model = _resolve_dynamics_model(
            algo, agent_cfg, is_classic, skrl_env.device, dynamics_model
        )
        if dynamics_model is not None and hasattr(raw_env, "set_dynamics_model"):
            raw_env.set_dynamics_model(dynamics_model)

        get_f_and_B, get_rollout = _env_attrs(raw_env, "get_f_and_B", "get_rollout")
        trainer_cfg = copy.deepcopy(cfg.get("trainer", {}))
        trainer_cfg.pop("class", None)
        trainer_cfg.setdefault("close_environment_at_exit", False)

        # C2RL keeps its contraction-metric (cm), CMG-synthesis (cmg), and
        # empirical-dynamics knobs in their own top-level yaml categories (not
        # nested under agent:) — see _setup_c2rl, which merges these into
        # agent_cfg before building the C2RLAgent. Other contraction algos
        # (C3M/LQR/SD-LQR) don't have these sections, so this is a no-op {} for
        # them.
        cm_cfg = copy.deepcopy(cfg.get("cm", {}))
        cmg_cfg = copy.deepcopy(cfg.get("cmg", {}))
        empirical_dynamics_cfg = copy.deepcopy(cfg.get("empirical_dynamics", {}))

        # use_empirical_dynamics only means anything to the algorithms that own a
        # dynamics model (C3M/C2RL). lqr/sdlqr/cvstem-lqr read get_f_and_B
        # directly, so the key is dropped here rather than reaching their cfg
        # filter — which would warn "not applied to the algorithm" on every run
        # (train.py sets it for every contraction algorithm).
        if algo in _ANALYTIC_CONTROLLERS:
            agent_cfg.pop("use_empirical_dynamics", None)
        else:
            # Analytical dynamics (use_empirical_dynamics=False) needs the env's exact
            # get_f_and_B, which only classic envs expose. Isaac envs must learn a
            # NeuralDynamics (use_empirical_dynamics=True) — train.py forces this, so
            # this guard only fires on a hand-rolled/standalone misconfiguration.
            use_empirical_dynamics = agent_cfg.get(
                "use_empirical_dynamics", empirical_dynamics_cfg.get("use_empirical_dynamics", False)
            )
            if not use_empirical_dynamics and not is_classic:
                raise ValueError(
                    "Analytical dynamics (use_empirical_dynamics=False) is only valid for "
                    "classic envs that expose an analytical get_f_and_B. Isaac Sim envs have "
                    "no analytical dynamics — set use_empirical_dynamics=True (pass "
                    "--use_empirical_dynamics) and let the agent train NeuralDynamics online."
                )

        # angle_idx: indices within an x-block that hold a raw (wrapping) angle.
        # Every network built below embeds these as (cos, sin) at its input —
        # see angle_utils.py / models.py — while the env/loss/error math keeps
        # using the raw state. Defaults to [] (no angle dims) for envs that
        # don't declare it.
        x_dim, u_dim, angle_idx = _env_attrs(raw_env, "x_dim", "u_dim", "angle_idx")
        angle_idx = list(angle_idx or [])
        sym = _resolve_symmetry(raw_env, x_dim, angle_idx)

        models_cfg = copy.deepcopy(cfg.get("models", {}))
        memory_cfg = copy.deepcopy(cfg.get("memory", {}))

        # Snapshot of the raw (pre dataclass-filter) YAML dicts, taken before
        # any _setup_* mutates agent_cfg/trainer_cfg/models_cfg in place — see
        # contraction_metrics.log_raw_config for why this is logged separately
        # from skrl's own dataclass-based wandb config.
        raw_cfg_snapshot = copy.deepcopy({
            "agent": agent_cfg, "trainer": trainer_cfg, "models": models_cfg,
            "cm": cm_cfg, "cmg": cmg_cfg, "empirical_dynamics": empirical_dynamics_cfg,
        })

        if algo in ("c3m",):
            self._setup_c3m(skrl_env, device, obs_space, state_space, act_space,
                            agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim, u_dim,
                            angle_idx=angle_idx, sym=sym, raw_cfg_snapshot=raw_cfg_snapshot)
        elif algo in ("lqr", "sdlqr"):
            # LQR/SD-LQR build no networks (get_f_and_B is either analytical or
            # an externally-loaded NeuralDynamics whose angle_idx was already
            # baked in when it was trained) — nothing here to embed.
            self._setup_sdlqr(skrl_env, device, obs_space, state_space, act_space,
                              agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr=(algo == "lqr"), x_dim=x_dim, u_dim=u_dim, angle_idx=angle_idx, sym=sym,
                              raw_cfg_snapshot=raw_cfg_snapshot)
        elif algo in ("cvstem-lqr",):
            # CV-STEM-LQR reads the same `cm:`/`cmg:` blocks C2RL does (joint SDP
            # + metric-network knobs), merged into agent_cfg. The agent builds and
            # freezes its own CholMetric — no model comes from `models:`.
            #
            # The joint SDP samples uniformly over the env's own state box and
            # differences W̄ against I over its own dt — both read from the env,
            # never configured (see CVSTEMLQRAgent).
            x_min, x_max, env_dt = _env_attrs(raw_env, "X_MIN", "X_MAX", "dt")
            self._setup_cvstem_lqr(skrl_env, device, obs_space, state_space, act_space,
                                   agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B,
                                   x_dim=x_dim, u_dim=u_dim, angle_idx=angle_idx, sym=sym,
                                   raw_cfg_snapshot=raw_cfg_snapshot,
                                   cm_cfg=cm_cfg, cmg_cfg=cmg_cfg,
                                   x_lo=x_min, x_hi=x_max, dt=env_dt)
        elif algo in ("c2rl-ppo", "c2rl-sac"):
            # base_algorithm is derived from the algo string itself (i.e. which
            # entry point / yaml you used), not a config toggle — see
            # C2RLAgent's base_algorithm constructor kwarg.
            self._setup_c2rl(skrl_env, device, obs_space, state_space, act_space,
                             agent_cfg, trainer_cfg, models_cfg, memory_cfg, get_rollout, get_f_and_B, x_dim, u_dim,
                             base_algorithm=("PPO" if algo == "c2rl-ppo" else "SAC"), angle_idx=angle_idx, sym=sym,
                             raw_cfg_snapshot=raw_cfg_snapshot,
                             cm_cfg=cm_cfg, cmg_cfg=cmg_cfg, empirical_dynamics_cfg=empirical_dynamics_cfg)

    def _setup_c3m(self, env, device, obs_space, state_space, act_space,
                   agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None,
                   angle_idx=None, sym=None, raw_cfg_snapshot=None):
        angle_idx = list(angle_idx or [])
        from c2rl.agents.skrl.c3m import C3MAgent, C3MCfg, C3MSkrlTrainer, C3MTrainerCfg

        models = _build_c3m_models(models_cfg, agent_cfg, obs_space, act_space, device, x_dim, u_dim, angle_idx, sym)
        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)

        agent = C3MAgent(
            cfg=C3MCfg(**_filter_cfg_fields(agent_cfg, C3MCfg, context="ContractionRunner agent")),
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
        )
        trainer_cfg.setdefault("timesteps", 30000)
        trainer = C3MSkrlTrainer(
            cfg=C3MTrainerCfg(**_filter_cfg_fields(trainer_cfg, C3MTrainerCfg, context="ContractionRunner trainer")),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._trainer._wandb_raw_cfg = raw_cfg_snapshot
        self._env = env
        self._mode = "c3m"

        # C3M has no discounting (no_op discount_factor=None) — its certificate
        # is a hard per-step LMI, not a discounted-objective approximation.
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent_cfg.get("lbd"),
                discount_factor=None,
                static_metric_bounds=(1.0 / w_lb, 1.0 / w_ub),
                cmg_bounds_fn=_make_cmg_bounds_fn(models["cmg"], w_lb),
            )

    def _setup_sdlqr(self, env, device, obs_space, state_space, act_space,
                     agent_cfg, trainer_cfg, models_cfg, get_f_and_B, lqr: bool = False, x_dim=None, u_dim=None, angle_idx=None, sym=None,
                     raw_cfg_snapshot=None):
        import dataclasses

        from skrl.trainers.torch import SequentialTrainer
        from skrl.trainers.torch.base import TrainerCfg

        from c2rl.agents.skrl.sdlqr import LQRAgent, LQRCfg, SDLQRAgent, SDLQRCfg

        cfg_cls = LQRCfg if lqr else SDLQRCfg
        agent_cls = LQRAgent if lqr else SDLQRAgent

        agent = agent_cls(
            cfg=cfg_cls(**_filter_cfg_fields(agent_cfg, cfg_cls, context="ContractionRunner agent")),
            models={},
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_f_and_B=get_f_and_B,
            x_dim=x_dim,
            u_dim=u_dim,
            angle_idx=angle_idx,
        )
        # SequentialTrainer for eval — no gradient updates. Pass trainer_cfg
        # through in full (like C3M/C2RL below) rather than a hardcoded
        # {timesteps, close_environment_at_exit} dict — the latter silently
        # dropped `headless` (set by train.py for every classic run), so
        # SequentialTrainer.train() called self.env.render() every step and
        # crashed with NotImplementedError (classic BaseEnv has no render()).
        tcfg = dataclasses.asdict(TrainerCfg(**_filter_cfg_fields({
            **trainer_cfg,
            "timesteps": trainer_cfg.get("timesteps", 10000),
            "close_environment_at_exit": False,
        }, TrainerCfg, context="ContractionRunner trainer")))
        # SequentialTrainer.__init__ already calls agent.init() (and thus
        # wandb.init(), if enabled) synchronously — unlike the other
        # algorithms' project-owned trainers, there's no later train()-time
        # hook to defer to, so push the raw config right here.
        trainer = SequentialTrainer(cfg=tcfg, env=env, agents=agent)
        from c2rl.agents.skrl.contraction_metrics import log_raw_config
        log_raw_config(raw_cfg_snapshot)
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "lqr" if lqr else "sdlqr"

    def _setup_cvstem_lqr(self, env, device, obs_space, state_space, act_space,
                          agent_cfg, trainer_cfg, models_cfg, get_rollout, get_f_and_B,
                          x_dim=None, u_dim=None, angle_idx=None, sym=None, raw_cfg_snapshot=None,
                          cm_cfg=None, cmg_cfg=None, x_lo=None, x_hi=None, dt=None):
        """Build a CVSTEMLQRAgent — Tsukamoto's CV-STEM/NCM contraction controller.

        Analytical (like SD-LQR), so a SequentialTrainer / no gradient updates.
        The `cm:`/`cmg:` yaml blocks (joint-SDP + metric-network knobs, kept out
        of `agent:` exactly as for C2RL) are merged into agent_cfg so
        CVSTEMLQRCfg sees them as one flat namespace. The agent owns its metric
        network (nn_modules.CholMetric, regressed onto chol(M)), so nothing is
        built from `models:` here."""
        import dataclasses

        from skrl.trainers.torch import SequentialTrainer
        from skrl.trainers.torch.base import TrainerCfg

        from c2rl.agents.skrl.cvstem_lqr import CVSTEMLQRAgent, CVSTEMLQRCfg

        # agent_cfg last so CLI overrides survive — see _setup_c2rl's identical merge.
        agent_cfg = {**(cm_cfg or {}), **(cmg_cfg or {}), **agent_cfg}
        angle_idx = list(angle_idx or [])

        agent = CVSTEMLQRAgent(
            cfg=CVSTEMLQRCfg(**_filter_cfg_fields(agent_cfg, CVSTEMLQRCfg, context="ContractionRunner agent")),
            models={},
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_f_and_B=get_f_and_B,
            x_lo=x_lo,
            x_hi=x_hi,
            dt=dt,
            x_dim=x_dim,
            u_dim=u_dim,
            angle_idx=angle_idx,
        )
        # Same fix as _setup_lqr_sdlqr above — pass trainer_cfg through in
        # full so `headless` (train.py) isn't silently dropped.
        tcfg = dataclasses.asdict(TrainerCfg(**_filter_cfg_fields({
            **trainer_cfg,
            "timesteps": trainer_cfg.get("timesteps", 10000),
            "close_environment_at_exit": False,
        }, TrainerCfg, context="ContractionRunner trainer")))
        trainer = SequentialTrainer(cfg=tcfg, env=env, agents=agent)
        from c2rl.agents.skrl.contraction_metrics import log_raw_config
        log_raw_config(raw_cfg_snapshot)
        self._agent = agent
        self._trainer = trainer
        self._env = env
        self._mode = "cvstem-lqr"

        # Certificate for the PathTracking plot — like C3M, this is a hard
        # per-step contraction controller (no discounting). The metric envelope
        # is not configured here: the joint SDP leaves ν free and returns it, and
        # W̄ ∈ [I, χI] with W = W̄/ν gives M ∈ [ν/χ, ν] exactly.
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent.metric_lbd,       # the linesearch may have chosen it, not the yaml
                discount_factor=None,
                static_metric_bounds=(agent.metric_nu, agent.metric_nu / agent.metric_chi),
                cmg_bounds_fn=None,
            )

    def _setup_c2rl(self, env, device, obs_space, state_space, act_space,
                    agent_cfg, trainer_cfg, models_cfg, memory_cfg, get_rollout, get_f_and_B, x_dim=None, u_dim=None,
                    base_algorithm: str = "PPO", angle_idx=None, sym=None, raw_cfg_snapshot=None,
                    cm_cfg=None, cmg_cfg=None, empirical_dynamics_cfg=None):
        """Build a single-policy C2RLAgent. Always builds a CMG network
        (MetricModel), synthesized offline before Phase B and then frozen
        (Tsukamoto NCM) — ``cmg_method`` (yaml `cm:` block) selects how: "ccm"
        (default) trains it directly via C1/C2 loss minimization; "cvstem" solves
        a per-state SDP and MSE-regresses the network onto the solutions. The CMG
        is always built as a ``BoundedCCM_Generator`` (``constrain_eigenvalues``
        forced ``True`` here, regardless of yaml) — see c2rl.py's module
        docstring for why.

        ``cm_cfg``/``cmg_cfg``/``empirical_dynamics_cfg`` are the yaml's
        top-level ``cm:``/``cmg:``/``empirical_dynamics:`` blocks — merged into
        ``agent_cfg`` here so C2RLAgent (which reads its full raw cfg dict as
        one flat namespace, see ``C2RLPPOCfg``/``C2RLSACCfg``) doesn't need to
        know they came from separate yaml sections. Kept out of ``agent:`` in
        the yaml purely so the contraction-metric / CMG-synthesis /
        empirical-dynamics knobs each read as their own config unit.
        """
        angle_idx = list(angle_idx or [])
        from c2rl.agents.skrl.c2rl import C2RLAgent, C2RLSkrlTrainer, C2RLTrainerCfg

        # agent_cfg last so it wins: train.py injects CLI overrides (e.g.
        # --use_empirical_dynamics) into the `agent:` block, and merging the
        # yaml category blocks after it silently clobbered them with the yaml
        # default. No yaml declares the same key in two blocks, so for a
        # yaml-only run the merge order makes no difference.
        agent_cfg = {**(cm_cfg or {}), **(cmg_cfg or {}), **(empirical_dynamics_cfg or {}), **agent_cfg}

        base_algorithm = base_algorithm.upper()

        # Phase B's memory: PPO's is exactly one rollout epoch (sized by
        # agent.rollouts); SAC's is a persistent replay buffer, sized via the
        # yaml's memory.memory_size (same convention as C3M — see C2RLAgent).
        mem_size = memory_cfg.get("memory_size", -1)
        rollouts = agent_cfg.get("rollouts", 300)
        if base_algorithm == "PPO" and mem_size not in (-1, rollouts):
            raise ValueError(
                f"[C2RL] memory.memory_size={mem_size} must be -1 (automatically "
                f"determined, same as agent:rollouts) or exactly match agent.rollouts "
                f"({rollouts}) for PPO."
            )
        agent_cfg["memory_size"] = mem_size

        w_lb = agent_cfg.get("w_lb", 0.1)
        w_ub = agent_cfg.get("w_ub", 10.0)
        models = {}

        # Always build a CMG network (MetricModel), synthesized offline (Phase A
        # — see C2RLAgent.synthesize_cmg) and frozen before Phase B. Forced
        # constrain_eigenvalues=True regardless of yaml — see c2rl.py's module
        # docstring for why (C2RLAgent also guards this at construction time).
        from c2rl.agents.skrl.models import MetricModel
        cmg_net = models_cfg.get("cmg", {}).get("network", [{}])
        cmg_hd = cmg_net[0].get("layers", [256, 256]) if cmg_net else [256, 256]
        cmg_act = cmg_net[0].get("activations", "tanh") if cmg_net else "tanh"
        # cmg_method decides which matrix the head emits: "cvstem" -> M
        # directly (its SDP targets invert once offline, and the reward wants
        # M, so every env step drops a batched SPD inverse); "ccm" -> W, whose
        # C1/C2 losses are written in W. See BoundedCCM_Generator.
        _cmg_method = str(agent_cfg.get("cmg_method", "cvstem")).lower()
        cmg_model = MetricModel(
            obs_space, act_space, device, hidden_dim=cmg_hd, activation=cmg_act,
            x_dim=x_dim, angle_idx=angle_idx, sym=_cmg_symmetry(sym), constrain_eigenvalues=True, w_lb=w_lb, w_ub=w_ub,
            outputs_metric=(_cmg_method == "cvstem"),
        )
        models["cmg"] = cmg_model

        # NeuralDynamics (learned ẋ = f(x) + B(x)·u) — feeds Phase A's CMG
        # synthesis. Built only when learning dynamics (use_empirical_dynamics=
        # True); Isaac envs' sole f/B source, classic envs default to analytical.
        if "dynamics" in models_cfg and agent_cfg.get("use_empirical_dynamics", False):
            from c2rl.agents.skrl.nn_modules import NeuralDynamics
            dyn_net = models_cfg["dynamics"].get("network", [{}])[0]
            dyn_hd = dyn_net.get("layers", [256, 256])
            dyn_act = dyn_net.get("activations", "relu")
            models["dynamics"] = NeuralDynamics(x_dim, u_dim, hidden_dim=dyn_hd, activation=dyn_act, device=device, angle_idx=angle_idx, sym=_cmg_symmetry(sym))

        # Phase B deployed policy + critic — control/mlp backbone dispatch, same
        # as C3M's. Value (PPO) / twin-Q + targets (SAC) see the same
        # angle-bearing x/xref blocks as the policy and embed them identically.
        models["policy"] = _build_gaussian_policy(models_cfg, "policy", obs_space, state_space, act_space,
                                                  device, x_dim, angle_idx, agent_class=f"C2RL-{base_algorithm}",
                                                  sym=sym)
        models.update(_build_critics(models_cfg, "critic", base_algorithm, obs_space, act_space,
                                     device, x_dim, angle_idx, key_prefix="", sym=sym,
                                     state_space=state_space,
                                     critic_encoder=models_cfg.get("critic", {}).get("encoder"),
                                     critic_combine=models_cfg.get("critic", {}).get("combine", "concat"),
                                     critic_embed_dim=int(models_cfg.get("critic", {}).get("embed_dim", 64)),
                                     critic_encoder_stride=int(models_cfg.get("critic", {}).get("encoder_stride", 1)),
                                     critic_analytic_potential=bool(
                                         models_cfg.get("critic", {}).get("analytic_potential", False)),
                                     w_lb=float(agent_cfg.get("cm", {}).get("w_lb", 0.01))))

        # Pass the raw agent_cfg dict (not a pre-parsed C2RLPPOCfg/C2RLSACCfg) —
        # C2RLAgent keeps the full dict in self._raw_cfg, which make_base_rl_cfg()
        # later filters against PPO_CFG/SAC_CFG. Pre-parsing first would silently
        # drop any PPO/SAC-specific field not also declared on the matching C2RL
        # cfg dataclass (this bit us for base_algorithm before it became an
        # explicit constructor kwarg — see C2RLAgent.__init__).
        agent = C2RLAgent(
            cfg=agent_cfg,
            models=models,
            observation_space=obs_space,
            state_space=state_space,
            action_space=act_space,
            device=device,
            get_rollout=get_rollout,
            get_f_and_B=get_f_and_B,
            base_algorithm=base_algorithm,
            x_dim=x_dim,
            u_dim=u_dim,
            num_envs=env.num_envs,
            angle_idx=angle_idx,
        )
        trainer_cfg.setdefault("timesteps", 300000)
        trainer = C2RLSkrlTrainer(
            cfg=C2RLTrainerCfg(**_filter_cfg_fields(trainer_cfg, C2RLTrainerCfg, context="ContractionRunner trainer")),
            env=env,
            agents=agent,
        )
        self._agent = agent
        self._trainer = trainer
        self._trainer._wandb_raw_cfg = raw_cfg_snapshot
        self._env = env
        self._mode = "c2rl"

        # Single deployed (Phase B) policy. The PathTracking figure's theoretical
        # bound inflates by 1/(1-discount_factor); the plotted certificate reads
        # the CMG's per-state metric bounds (cmg_bounds_fn), falling back to the
        # static [1/w_lb, 1/w_ub] only if the env doesn't wire cmg_bounds_fn up.
        if hasattr(env, "set_contraction_certificate"):
            env.set_contraction_certificate(
                agent_cfg.get("lbd"),
                discount_factor=agent_cfg.get("discount_factor", 0.99),
                static_metric_bounds=(1.0 / w_lb, 1.0 / w_ub),
                cmg_bounds_fn=_make_cmg_bounds_fn(cmg_model, w_lb),
            )

    # ── public interface ────────────────────────────────────────────────── #

    @property
    def agent(self):
        if self._mode == "skrl":
            return self._runner.agent
        return self._agent

    @property
    def trainer(self):
        if self._mode == "skrl":
            return getattr(self._runner, "_trainer", None) or getattr(self._runner, "trainer", None)
        return self._trainer

    def load(self, path: str) -> None:
        if self._mode == "skrl":
            self._runner.agent.load(path)
        else:
            self._agent.load(path)

    def run(self):
        if self._mode == "skrl":
            return self._runner.run()
        return self._trainer.train()
