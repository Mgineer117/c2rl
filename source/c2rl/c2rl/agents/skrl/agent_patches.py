"""Generic post-construction patches for skrl PPO/SAC agent instances.

Used by train.py (patches ``runner.agent``) and c2rl.py (patches its inner
PPO/SAC sub-agent; C2RL's outer agent has no ``.policy``/``.scheduler``). Each
patch no-ops if the agent lacks what it needs, so calling any of them on any
agent is safe.

Invariant: every call site patches before ``agent.init()`` (where skrl
allocates memory tensors). ``patch_caps_regularizer`` relies on it.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


def patch_kl_logging(agent) -> None:
    """Log per-epoch approximate KL to 'Policy / KL divergence'.

    skrl's PPO computes KL to drive KLAdaptiveLR but never records it, so
    kl_threshold early stops are invisible — even though they deflate that
    update's averaged losses (skrl divides by the full
    learning_epochs*mini_batches regardless of where it broke).

    ``KLAdaptiveLR.step(kl)`` is the only place the KL escapes skrl's
    ``_update`` — it is a local there and nothing else reads it (see skrl
    ppo.py: ``kl_divergences`` is built in the minibatch loop and consumed only
    under ``isinstance(self.scheduler, KLAdaptiveLR)``). So the metric exists
    only if that scheduler is attached.

    Rather than warn and drop the curve, this now attaches a KLAdaptiveLR that
    cannot move the learning rate — ``min_lr = max_lr = the optimizer's current
    lr`` — whenever a PPO-family agent has none. The scheduler clamps its own
    output to [min_lr, max_lr], so training is numerically identical to running
    without a scheduler while the KL hook fires every update.

    This is deliberately done in code rather than per-yaml: the metric is a
    diagnostic every PPO/C2RL-PPO run should have, and making it depend on
    remembering a config key is how it went missing in the first place
    (``learning_rate_scheduler: null`` silently dropped it).
    """
    import skrl.resources.schedulers.torch as _sched

    scheduler = getattr(agent, "scheduler", None)
    if isinstance(scheduler, _sched.KLAdaptiveLR):
        pass
    elif hasattr(agent, "scaler"):
        # A PPO/SAC-family agent: it runs the update loop that computes the KL.
        # Attach a pinned KLAdaptiveLR so the hook exists. SAC-family agents have
        # no `optimizer` of this shape and never build kl_divergences, so they
        # fall through to the no-op below rather than getting a dead scheduler.
        opt = getattr(agent, "optimizer", None)
        if opt is None or not getattr(opt, "param_groups", None):
            return
        lr = float(opt.param_groups[0]["lr"])
        try:
            scheduler = _sched.KLAdaptiveLR(opt, min_lr=lr, max_lr=lr,
                                            kl_threshold=getattr(agent.cfg, "kl_threshold", 0.008) or 0.008)
        except Exception as exc:  # noqa: BLE001 — never break training for a metric
            print(f"[patch] could not attach a pinned KLAdaptiveLR to "
                  f"{type(agent).__name__} ({exc}); 'Policy / KL divergence' "
                  f"will not be logged.")
            return
        agent.scheduler = scheduler
        print(f"[patch] {type(agent).__name__}: attached a PINNED KLAdaptiveLR "
              f"(min_lr = max_lr = {lr:g}) purely so 'Policy / KL divergence' is "
              f"logged; the learning rate cannot move.")
    else:
        # A wrapper agent (C2RL's outer agent has no .policy/.scaler of its own)
        # — it never computes a KL, so there is nothing to miss. Both train_utils
        # and C2RL call this, the latter on the inner sub-agent that does; warning
        # here would fire on every C2RL run and train everyone to ignore the real one.
        return

    _orig_step = scheduler.step

    def _step(kl=None, *, epoch=None):
        if kl is not None:
            agent.track_data("Policy / KL divergence", float(kl))
        _orig_step(kl, epoch=epoch)

    scheduler.step = _step


def patch_ppo_diagnostics(agent, *, disable_advantage_norm: bool = False) -> None:
    """Phase-0 collapse diagnostics. Swaps skrl's module-level ``compute_gae``
    for an instrumented copy of the same math (loop verbatim from skrl 1.4),
    logging per update::

        Diagnostics / raw advantage mean/std — pre-normalization GAE. Near-zero
            std means skrl's (adv - mean)/(std + 1e-8) manufactures a
            full-scale confident gradient out of noise — the mechanism behind
            "near-optimal policy -> destabilizing update".
        Diagnostics / critic explained variance — 1 - Var(returns-values)/
            Var(returns). Near 0 = critic has no predictive content yet; the
            first update where it jumps off 0 is the collapse candidate.

    ``disable_advantage_norm=True`` (Ablation C) also skips the normalization.

    Scope: ``compute_gae`` is a bare module-level name with no per-instance
    hook, so this patches it globally — fine while only one PPO agent runs per
    process, true of every entrypoint here. Idempotent; no-op for SAC.
    """
    # PPO.__init__ (already run, see module docstring) sets self.value; SAC has
    # critic_1/critic_2 instead. self._tensors_names would be the better check
    # but isn't set until agent.init(), which runs after this.
    if not hasattr(agent, "value"):
        return
    import skrl.agents.torch.ppo.ppo as _ppo_mod

    if getattr(_ppo_mod, "_c2rl_diagnostics_patched", False):
        return
    _ppo_mod._c2rl_diagnostics_patched = True

    def _instrumented_compute_gae(
        *,
        rewards,
        terminated,
        truncated,
        values,
        last_values,
        discount_factor: float = 0.99,
        lambda_coefficient: float = 0.95,
        time_limit_bootstrap: bool = False,
    ):
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
        memory_size = rewards.shape[0]
        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            advantage = (
                rewards[i] - values[i]
                + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
            )
            advantages[i] = advantage
        returns = advantages + values

        agent.track_data("Diagnostics / raw advantage mean", advantages.mean().item())
        agent.track_data("Diagnostics / raw advantage std", advantages.std().item())
        returns_var = returns.var()
        if returns_var > 1e-12:
            ev = 1.0 - (returns - values).var() / returns_var
            agent.track_data("Diagnostics / critic explained variance", ev.item())

        if disable_advantage_norm:
            return returns, advantages
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    _ppo_mod.compute_gae = _instrumented_compute_gae


def patch_sac_entropy_clamp(agent, min_log_alpha: float = -5.0, max_log_alpha: float = 2.0) -> None:
    """Clamp log_entropy_coefficient in-place after every entropy optimizer step.

    skrl's SAC applies grad_norm_clip to the policy/critic optimizers but not to
    entropy_optimizer, and alpha = exp(log_alpha) is unbounded. A noisy critic
    can push this one scalar's gradient large, and exp() turns a moderate
    excursion into a runaway alpha that dominates both the critic target and the
    policy loss — a textbook SAC divergence. Bounds alpha to ~[0.0067, 7.39],
    mirroring skrl's own clip_log_std. No-op without learn_entropy.
    """
    entropy_optimizer = getattr(agent, "entropy_optimizer", None)
    log_alpha = getattr(agent, "log_entropy_coefficient", None)
    if entropy_optimizer is None or log_alpha is None:
        return

    _orig_step = entropy_optimizer.step

    def _step(*args, **kwargs):
        result = _orig_step(*args, **kwargs)
        with torch.no_grad():
            log_alpha.clamp_(min_log_alpha, max_log_alpha)
        return result

    entropy_optimizer.step = _step


def patch_ppo_std_annealing(agent, std_dev_annealing: bool, kwargs: dict | None = None) -> None:
    """Adds manual standard deviation annealing to SKRL's PPO policy.

    When on: disables the entropy loss (entropy_loss_scale=0.0) and anneals
    log_std_parameter from its initial value to `final_log_std` over training.

    `std_dev_annealing` is not a yaml flag — callers derive it. train.py: from
    whether the backbone is in ``runner.CONTROL_BACKBONES``. c2rl.py: from
    ``base_algorithm == "PPO"``, i.e. always on for PPO, always off for SAC
    (which tunes log_std via automatic entropy tuning). Freezes
    log_std_parameter here; no-ops for a state-dependent log_std head
    (``mlp-squashed``). Only the schedule is yaml-configurable::

        agent:
          std_dev_annealing_kwargs:
            schedule: exponential   # linear | exponential | cosine
            final_log_std: -2.3     # target log_std (std ~= 0.1)
            power: 5.0              # exponential schedule only: progress**power

    Schedules (p = timestep/timesteps in [0, 1]):
      linear:       log_std = init + p * (final - init)
      exponential:  log_std = init + p**power * (final - init)  — slow early,
                    fast late; keeps exploration wide for most of training
      cosine:       log_std = init + (1 - cos(pi*p))/2 * (final - init)
    """
    if not std_dev_annealing:
        return
    kwargs = dict(kwargs or {})
    schedule = str(kwargs.pop("schedule", "linear")).lower()
    final_log_std = float(kwargs.pop("final_log_std", -2.0))
    power = float(kwargs.pop("power", 5.0))
    if kwargs:
        from skrl import logger
        logger.warning(f"std_dev_annealing_kwargs: ignoring unknown keys {sorted(kwargs)}")

    # Ignore entropy: annealing and entropy bonus fight each other
    if hasattr(agent, "_cfg") and isinstance(agent._cfg, dict):
        agent._cfg["entropy_loss_scale"] = 0.0
    if hasattr(agent, "cfg"):
        if isinstance(agent.cfg, dict):
            agent.cfg["entropy_loss_scale"] = 0.0
        else:
            setattr(agent.cfg, "entropy_loss_scale", 0.0)

    if not hasattr(agent, "policy") or not hasattr(agent.policy, "log_std_parameter"):
        return

    # Disable gradients on log_std_parameter because we update it manually
    agent.policy.log_std_parameter.requires_grad_(False)

    # Captured lazily on the first post_interaction, not here: this patch runs
    # at __init__, before C2RL's Phase A + pretraining. Capturing eagerly froze
    # "initial" at the yaml's initial_log_std, so a post-pretrain override
    # (residual_pretrain_init_log_std) got silently reverted on the next
    # post_interaction. Lazy = "initial" means what Phase B actually starts at.
    initial_log_std = None

    def _ratio(p: float) -> float:
        if schedule == "exponential":
            return p ** power
        if schedule == "cosine":
            return (1.0 - math.cos(math.pi * p)) / 2.0
        return p  # linear

    _orig_post = agent.post_interaction

    def _annealed_post(*, timestep: int, timesteps: int) -> None:
        nonlocal initial_log_std
        if initial_log_std is None:
            initial_log_std = agent.policy.log_std_parameter.mean().item()
        progress = min(1.0, max(0.0, timestep / max(1, timesteps)))
        current_log_std = initial_log_std + _ratio(progress) * (final_log_std - initial_log_std)
        agent.policy.log_std_parameter.data.fill_(current_log_std)
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _annealed_post


def _resolve_x_dim(policy) -> int | None:
    """Length of the leading ``x`` block of a ``[x, xref, uref]`` observation,
    or ``None`` for a flat observation with no such split.

    Every backbone knows this number but spells it differently: residual/
    squashed MLPs use ``_x_dim``; CLActorModel/SquashedCLActorModel keep it only
    on their ``cl_actor`` submodule. Missing that last case silently returned
    None for exactly the two backbones where the x-only restriction is
    load-bearing (see patch_caps_regularizer on uref pass-through).
    """
    for owner, attr in ((policy, "x_dim"), (policy, "_x_dim"),
                        (getattr(policy, "cl_actor", None), "x_dim")):
        x_dim = getattr(owner, attr, None) if owner is not None else None
        if x_dim:
            return int(x_dim)
    return None


def patch_caps_regularizer(
    agent,
    *,
    temporal_scale: float = 0.0,
    spatial_scale: float = 0.0,
    spatial_std: float = 0.05,
    batch_size: int = 1024,
) -> None:
    """Add CAPS action-smoothness regularization to the policy loss.

    CAPS (Mysore et al. 2021) penalizes two kinds of non-smoothness in the
    policy mean — never the sampled action, whose noise we don't want to
    suppress and whose gradient would fight std annealing / entropy tuning::

        L_T = temporal_scale * || pi(s_t) - pi(s_{t+1}) ||^2   (chatter in time)
        L_S = spatial_scale  * || pi(s)   - pi(s_bar)   ||^2   (high state gain)
                                        s_bar ~ N(s, spatial_std^2)

    Loss, not reward. A ``-||u_t - u_{t-1}||^2`` reward makes the return depend
    on u_{t-1}, which isn't observed — a different, partially observed MDP the
    critic can only model as noise. Adding u_{t-1} to the obs changes obs_dim
    and trips the ``obs_dim == 2*x_dim + u_dim`` assertion the CLActor backbones
    rely on. And L_S isn't expressible as a reward at all: the env has no handle
    on pi to evaluate at a second state. As a policy-loss term CAPS keeps the
    MDP/obs/dynamics identical, so the offline CV-STEM certificate stays valid.

    States come from the agent's own memory, so CAPS inherits each algorithm's
    data distribution rather than imposing a third: SAC's replay buffer, PPO's
    rollout memory (exactly the on-policy batch, nothing older).

    The (s_t, s_t+1) pairing needs a ``next_observations`` column that SAC
    allocates but PPO doesn't. Allocated unconditionally here: create_tensor is
    idempotent and the memory is still empty (see module docstring on ordering),
    so there's no per-algorithm branch to get wrong. Filled by wrapping the
    agent's existing ``add_samples`` — a second call would advance memory_index
    twice and desync the columns. Read via ``sample_by_index``, never by
    extending ``_tensors_names``: both agents unpack ``memory.sample()`` into a
    fixed-arity tuple, so one extra name would raise.

    Autoreset: a pair straddling an episode boundary isn't a real (s_t, s_t+1),
    and smoothness across a reset is a discontinuity the policy can't control.
    ``caps_valid`` drops both the ``terminated | truncated`` step (in-place
    autoreset, what both families here do) and the step after a done (what a
    next-step-autoreset env would flag), so it's correct under either convention
    without detecting which is live. Costs <=2 transitions/episode (<1% here).

    Injection point: both agents route every policy backward through
    ``scaler.scale(loss).backward()`` right after the update's one grad-enabled
    ``policy.act(role="policy")``. Arming on that act and consuming on the next
    scale puts the CAPS gradient in the same backward as the policy loss, hence
    inside grad_norm_clip — a separate ``.backward()`` would escape it. SAC's
    critic scale precedes the act and its entropy scale follows the consume, so
    neither is hit. PPO's kl_threshold break leaves the flag armed and it is
    consumed by the next update's policy scale — still never a critic/entropy one.

    Spatial perturbation covers only the leading ``x`` block when the policy
    exposes an x_dim (else the whole obs). For control backbones
    ``u = uref + feedback(x - xref)``, so perturbing uref shifts the output by
    exactly that amount — irreducible feedforward pass-through that would floor
    L_S and push the policy to suppress its own uref term.

    ``spatial_std`` is in raw obs units (perturbation added pre-preprocessor),
    so enabling use_state_norm would shrink what the policy actually sees.
    Inert today (state norm off everywhere), but note a single scalar sigma
    under-regularizes any dimension with a much wider range than the rest.

    No-op unless a scale is positive, or without policy/scaler/memory (C2RL's
    outer agent — it patches its inner sub-agent directly).
    """
    temporal_scale = float(temporal_scale)
    spatial_scale = float(spatial_scale)
    if temporal_scale <= 0.0 and spatial_scale <= 0.0:
        return
    policy = getattr(agent, "policy", None)
    scaler = getattr(agent, "scaler", None)
    memory = getattr(agent, "memory", None)
    if policy is None or scaler is None or memory is None:
        return

    device = getattr(memory, "device", None) or next(policy.parameters()).device
    x_dim = _resolve_x_dim(policy)

    # skrl's own "no preprocessor" fallback is _empty_preprocessor, which already
    # swallows the train= kwarg — so both branches are callable the same way.
    def _identity(t, **_kwargs):
        return t

    obs_pre = getattr(agent, "_observation_preprocessor", None) or _identity
    state_pre = getattr(agent, "_state_preprocessor", None) or _identity
    _orig_act = policy.act

    # ── make the (s_t, s_t+1) pairing readable from the agent's own memory ──
    # Idempotent, so a no-op for columns the agent allocates itself in init().
    # next_states allocates nothing when state_space is None (the norm here);
    # sample_by_index then yields None and the policy gets states=None, exactly
    # as in a normal update.
    memory.create_tensor(name="next_observations", size=agent.observation_space, dtype=torch.float32)
    memory.create_tensor(name="next_states", size=agent.state_space, dtype=torch.float32)
    memory.create_tensor(name="caps_valid", size=1, dtype=torch.bool)

    _names = ["observations", "next_observations", "caps_valid", "states", "next_states"]

    _prev_done = torch.zeros((memory.num_envs, 1), dtype=torch.bool, device=device)
    _pending: dict = {}

    _orig_add_samples = memory.add_samples

    def _add_samples(**tensors):
        if _pending:
            tensors.update(_pending)
            _pending.clear()
        return _orig_add_samples(**tensors)

    memory.add_samples = _add_samples

    _orig_record = agent.record_transition

    def _record(*, observations, states, actions, rewards, next_observations, next_states,
                terminated, truncated, infos, timestep, timesteps):
        done = terminated | truncated
        _pending["caps_valid"] = ~done & ~_prev_done   # see autoreset in the docstring
        _pending["next_observations"] = next_observations
        _pending["next_states"] = next_states
        _prev_done.copy_(done)
        return _orig_record(
            observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states,
            terminated=terminated, truncated=truncated, infos=infos,
            timestep=timestep, timesteps=timesteps,
        )

    agent.record_transition = _record

    # ── the CAPS loss itself ────────────────────────────────────────────────
    # _orig_act, not policy.act: the arming wrapper below must not see these
    # forwards, and calling the pre-patch method is what keeps it from doing so.
    def _policy_mean(obs, states):
        inputs = {"observations": obs_pre(obs, train=False),
                  "states": states if states is None else state_pre(states, train=False)}
        _, outputs = _orig_act(inputs, role="policy")
        return outputs["mean_actions"]

    def _caps_loss():
        size = len(memory)
        if size == 0:
            return None
        # sample_by_index rather than sample(): the latter also overwrites
        # memory.sampling_indexes, which belongs to the agent's own update.
        indexes = torch.randint(0, size, (min(batch_size, size),), device=device)
        obs, next_obs, valid, states, next_states = memory.sample_by_index(
            names=_names, indexes=indexes
        )[0]

        mean = _policy_mean(obs, states)
        loss = None

        if temporal_scale > 0.0 and bool(valid.any()):
            next_mean = _policy_mean(next_obs, next_states)
            # Masked mean over surviving pairs, not sum/N — else the effective
            # coefficient silently shrinks with the episode-boundary fraction,
            # swinging with the termination rate as the policy improves.
            sq = ((mean - next_mean) ** 2).sum(dim=-1, keepdim=True)
            l_t = (sq * valid).sum() / valid.sum().clamp(min=1)
            agent.track_data("Loss / CAPS temporal", l_t.item())
            loss = temporal_scale * l_t

        if spatial_scale > 0.0:
            noise = torch.randn_like(obs) * spatial_std
            if x_dim:
                noise[:, x_dim:] = 0.0  # see docstring: never perturb xref/uref
            bar_mean = _policy_mean(obs + noise, states)
            l_s = ((mean - bar_mean) ** 2).sum(dim=-1).mean()
            agent.track_data("Loss / CAPS spatial", l_s.item())
            loss = spatial_scale * l_s if loss is None else loss + spatial_scale * l_s

        return loss

    # ── arm on the update's policy forward, consume on the next scale() ─────
    armed = False

    def _act(inputs, *, role: str = ""):
        nonlocal armed
        out = _orig_act(inputs, role=role)
        if role == "policy" and torch.is_grad_enabled():
            armed = True
        return out

    policy.act = _act

    _orig_scale = scaler.scale

    def _scale(loss):
        nonlocal armed
        if armed:
            armed = False
            caps = _caps_loss()
            if caps is not None:
                loss = loss + caps
        return _orig_scale(loss)

    scaler.scale = _scale


def patch_algo_namespace(agent, algo_name: str) -> None:
    """Rewrite track_data keys "{cat} / {name}" -> "{cat} / {algo_name}/{name}".

    C3M/C2RL already namespace their internals this way so runs from different
    algorithms don't collide on the same wandb panels; standalone PPO/SAC never
    did, leaving skrl's own keys ("Loss / Policy loss", ...) un-namespaced.

    "Reward / *" is deliberately exempt: skrl's base Agent writes that exact key
    straight into tracking_data (bypassing track_data) to pick best_agent.pt, so
    renaming would silently break selection. Stability/*, Episode/*, Info/* also
    bypass track_data, so they are unaffected either way.
    """
    orig_track_data = agent.track_data

    def _wrapped(tag, value):
        if " / " in tag and not tag.startswith("Reward / "):
            category, name = tag.split(" / ", 1)
            if not name.startswith(f"{algo_name}/"):
                tag = f"{category} / {algo_name}/{name}"
        orig_track_data(tag, value)

    agent.track_data = _wrapped


def best_metric_for(algorithm: str) -> str:
    """Which stability metric an algorithm's best_agent.pt should track.

    c3m/cvstem-lqr — the two controllers that certify a metric directly — use
    ``contraction_score`` (λ/overshoot, higher better). Everything else (learned
    policies, lqr/sdlqr) uses AUC (lower better). Accepts either the hyphen or
    underscore spelling of the algorithm name.
    """
    norm = str(algorithm).lower().replace("_", "-")
    return "contraction_score" if norm in ("c3m", "cvstem-lqr") else "auc"


BEST_METRIC_KEY = "Checkpoint / best metric"


def patch_auc_checkpoint(agent, metric: str = "auc") -> None:
    """Override agent.post_interaction to pick best_agent.pt by ``metric``.

    skrl selects best_agent.pt by the highest ``Reward / Total reward (mean)``,
    read straight out of ``tracking_data`` in its ``post_interaction``. This
    used to be redirected by overwriting that key with the stability metric —
    which silently turned a reward panel into ``-AUC``: ``(mean)`` no longer
    measured the same quantity as the untouched ``(min)``/``(max)`` (still real
    episode returns), so a diverging run plotted its "mean" below its "min".

    Instead the metric now goes to its own :data:`BEST_METRIC_KEY` panel, and
    the best-module bookkeeping is done here against that value, mirroring
    skrl's own logic. Base skrl's reward-based selection is then neutralized by
    parking ``checkpoint_best_modules["reward"]`` at ``+inf`` so its
    ``reward > stored`` test can never overwrite the modules chosen here.
    ``write_checkpoint`` still runs from the base call and persists them.

    "contraction_score": ``Stability/contraction_score_mean``, higher better, so
    no sign flip. "auc": ``-Stability/auc_mean`` (or ``-Episode/auc`` for
    vel-tracking), flipped since lower is better. If the chosen metric isn't
    logged this step, no selection happens — no cross-fallback between the two.
    See ``best_metric_for`` for the algorithm→metric mapping.
    """
    import copy

    _orig_post = getattr(agent, "post_interaction", None)
    if _orig_post is None:
        return

    best = {"value": -float("inf")}

    def _current_metric():
        # Stability/* carries a "_mean" suffix; Episode/auc (vel-tracking) does
        # not — it comes from a different logging path.
        if metric == "contraction_score":
            score_list = agent.tracking_data.get("Stability/contraction_score_mean")
            return float(score_list[-1]) if score_list else None
        # Prioritize Stability/auc (path tracking) over Episode/auc (velocity tracking)
        score_list = agent.tracking_data.get("Stability/auc_mean") or agent.tracking_data.get("Episode/auc")
        return -float(score_list[-1]) if score_list else None

    def _metric_post(*, timestep: int, timesteps: int) -> None:
        value = _current_metric()
        if value is not None:
            agent.track_data(BEST_METRIC_KEY, value)

        # Mirror skrl's cadence: it updates best modules then writes them, both
        # gated on checkpoint_interval inside the base call below.
        ts = timestep + 1
        if ts > 1 and agent.checkpoint_interval > 0 and not ts % agent.checkpoint_interval:
            if value is not None and value > best["value"]:
                best["value"] = value
                agent.checkpoint_best_modules["timestep"] = ts
                agent.checkpoint_best_modules["saved"] = False
                agent.checkpoint_best_modules["modules"] = {
                    k: copy.deepcopy(agent._get_internal_value(v))
                    for k, v in agent.checkpoint_modules.items()
                }
        # Park the base's comparison value so its own reward-based selection can
        # never win. Must be set every call, not just on checkpoint steps.
        agent.checkpoint_best_modules["reward"] = float("inf")
        _orig_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = _metric_post


def patch_prune_checkpoints(agent) -> None:
    """Keep only the newest ``agent_<timestep>.pt``, alongside ``best_agent.pt``.

    skrl writes one full checkpoint every ``checkpoint_interval`` steps and
    never removes the previous ones — with the configs' ``checkpoint_interval:
    "auto"`` (= timesteps/10) that is 10 full snapshots per run, none of which
    are useful once the next one lands. Only the latest snapshot (to resume
    from) and ``best_agent.pt`` (selected by ``patch_auc_checkpoint``'s metric)
    are ever actually loaded, so the older snapshots are pure disk/inode cost.

    ``best_agent.pt`` is untouched: it does not match the ``agent_*.pt`` glob.
    """
    _orig_write = getattr(agent, "write_checkpoint", None)
    if _orig_write is None:
        return

    def _pruning_write(*, timestep: int, timesteps: int) -> None:
        _orig_write(timestep=timestep, timesteps=timesteps)
        # ponytail: assumes skrl's default whole-agent layout (store_separately
        # is unset repo-wide). If store_separately is ever turned on, the
        # per-module "<name>_<tag>.pt" files need the same prune-by-prefix.
        ckpt_dir = Path(getattr(agent, "experiment_dir", "")) / "checkpoints"
        snapshots = sorted(ckpt_dir.glob("agent_*.pt"), key=lambda p: p.stat().st_mtime)
        for stale in snapshots[:-1]:
            stale.unlink(missing_ok=True)

    agent.write_checkpoint = _pruning_write
