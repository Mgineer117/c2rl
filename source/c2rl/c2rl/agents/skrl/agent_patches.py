"""Generic post-construction patches for skrl PPO/SAC agent instances.

Used by train.py (patches ``runner.agent``) and c2rl.py (patches its inner
PPO/SAC sub-agent; C2RL's outer agent has no ``.policy``/``.scheduler``). Each
patch no-ops if the agent lacks what it needs, so calling any of them on any
agent is safe.

Invariant: every call site patches before ``agent.init()`` (where skrl
allocates memory tensors).
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
    # (a post-construction override of log_std) got silently reverted on the next
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
    load-bearing for the uref pass-through.
    """
    for owner, attr in ((policy, "x_dim"), (policy, "_x_dim"),
                        (getattr(policy, "cl_actor", None), "x_dim")):
        x_dim = getattr(owner, attr, None) if owner is not None else None
        if x_dim:
            return int(x_dim)
    return None




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


def patch_kl_logging_post_update(agent) -> None:
    """Log the KL between the policy the update ENDED on and the rollout policy.

    ``Policy / KL divergence`` is skrl's own number and stays as it is: the mean
    of the per-mini-batch KLs collected during one epoch, taken from the value
    handed to ``KLAdaptiveLR.step``. That mean is not the size of the step the
    update actually took. It mixes early mini-batches, evaluated when the policy
    had barely moved, with the one that tripped ``kl_threshold`` -- skrl appends
    the KL and only then breaks (ppo.py:398 then :401), so the tripping value is
    inside the average that gets logged. A logged 5.5 against a threshold of
    0.032 is therefore the signature of a correct trip, not a failed one, and it
    says nothing directly about where the policy came to rest.

    ``Policy / KL divergence (post-update)`` is that missing quantity: after the
    update returns, recompute ``log pi(a|s)`` for the whole rollout under the
    current parameters and compare it to the ``log_prob`` stored when those
    actions were taken. The tripping mini-batch's gradient is discarded
    (ppo.py's break precedes ``optimizer.step``), so "current parameters" is
    exactly the last applied iterate -- the policy the update ended on -- and
    the stored log_prob is the policy the rollout came from.

    Uses the same k3 estimator as skrl, ``(exp(r) - 1) - r``, so the two curves
    are directly comparable, and evaluates over the full rollout rather than one
    mini-batch, which removes the sampling noise that makes the per-mini-batch
    value swing by orders of magnitude.

    Costs one extra no-grad forward pass per update over ``rollouts x num_envs``
    transitions.
    """
    import torch

    memory = getattr(agent, "memory", None)
    policy = getattr(agent, "policy", None)
    if memory is None or policy is None or not hasattr(agent, "track_data"):
        return
    if getattr(agent, "_kl_post_update_logged", False):
        return

    _orig_update = agent.update

    def _update_then_log_kl(*, timestep: int, timesteps: int) -> None:
        _orig_update(timestep=timestep, timesteps=timesteps)
        try:
            obs = memory.get_tensor_by_name("observations")
            actions = memory.get_tensor_by_name("actions")
            old_log_prob = memory.get_tensor_by_name("log_prob")
        except (KeyError, AttributeError):
            return  # memory not laid out as PPO's; nothing to compare against
        with torch.no_grad():
            flat = {
                "observations": agent._observation_preprocessor(obs.view(-1, obs.shape[-1])),
                "taken_actions": actions.view(-1, actions.shape[-1]),
            }
            if hasattr(agent, "_state_preprocessor"):
                try:
                    states = memory.get_tensor_by_name("states")
                    flat["states"] = agent._state_preprocessor(states.view(-1, states.shape[-1]))
                except (KeyError, AttributeError):
                    pass
            _, outputs = policy.act(flat, role="policy")
            r = outputs["log_prob"].view(-1) - old_log_prob.view(-1)
            kl = ((torch.exp(r) - 1) - r).mean().item()
        agent.track_data("Policy / KL divergence (post-update)", kl)

    agent.update = _update_then_log_kl
    agent._kl_post_update_logged = True
