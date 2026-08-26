import os
import sys

import torch

_SAC_LIKE_ALGOS = {"sac", "c2rl-sac", "c2rl_sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_PPO_CLASSIC = 1024
_VEL_TASK_TO_ROBOT = {"Quadruped": "quadruped", "Humanoid": "humanoid", "Manipulator": "manipulator"}
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generated data lives under data/, not logs/. The distinction is lifetime and
# role, not format: logs/ holds the output of one run (checkpoints, tensorboard
# events, eval json) and is disposable per run, while data/ holds artifacts that
# are inputs to later runs of other algorithms — dynamics_data.npz (the
# reference trajectories path-tracking envs track) and the cm_data*.npz metric
# caches synthesized from it. Deleting logs/ must never force a re-synthesis.
_DATA_ROOT = os.path.join(_ROOT, "data")

def _default_num_envs_classic(algo: str) -> int:
    return _DEFAULT_NUM_ENVS_SAC if algo.lower() in _SAC_LIKE_ALGOS else _DEFAULT_NUM_ENVS_PPO_CLASSIC


def _run_metadata(args_cli, task: str) -> dict:
    """Identity of the run that produced an eval_results.json.

    The metrics in eval_results.json are anonymous on their own — the run
    directory is named by timestamp only, so nothing in the file says which
    seed/env/algorithm it came from. Multi-seed aggregation (run_seeds.sh +
    scripts/aggregate_seeds.py) needs exactly that to group runs, so it is
    written into the json rather than parsed back out of a path.

    CRL_RUN_TAG is set by run_seeds.sh to a per-launch tag, so the aggregator
    can select one batch of seeds instead of every historical run under logs/.
    """
    return {
        "task": task,
        "algorithm": getattr(args_cli, "algorithm", None),
        "seed": getattr(args_cli, "seed", None),
        "run_tag": os.environ.get("CRL_RUN_TAG", ""),
    }


def _resolve_symmetry_for_env(raw_env) -> int:
    """``_resolve_symmetry`` for the standalone PPO/SAC path, which has no
    ContractionRunner to ask. Delegates to the same verified check so both
    routes make the same call for the same env (never one quotiented and one
    not, which would make PPO and C2RL-PPO architecturally incomparable).
    """
    from c2rl.runners.contraction_runner import _env_attrs, _resolve_symmetry, _unwrap_env
    env = _unwrap_env(raw_env)
    # _env_attrs, not getattr: a classic SyncVectorEnv forwards none of its
    # sub-envs' attributes, so x_dim/angle_idx live one level deeper there (a
    # plain getattr silently returned None and disabled the quotient for the
    # standalone PPO/SAC route while ContractionRunner had it on -- which would
    # have made the PPO baseline and C2RL-PPO architecturally incomparable,
    # exactly the mismatch this function exists to prevent).
    x_dim, angle_idx = _env_attrs(env, "x_dim", "angle_idx")
    if x_dim is None:
        (x_dim,) = _env_attrs(env, "num_dim_x")
    return _resolve_symmetry(env, x_dim, list(angle_idx or []))


def _inject_angle_idx(agent_cfg: dict, angle_idx: list, sym=None) -> None:
    """Inject ``angle_idx``/``pos_dim`` into every model sub-block of agent_cfg["models"].

    Only the standalone PPO/SAC path needs this: those models are built by
    _gaussian_factory/_deterministic_factory (runner.py) purely from each
    yaml/cfg block's own keys, with no access to the env object. The
    ContractionRunner path (C3M/LQR/SDLQR/C2RL) is self-sufficient — it reads
    angle_idx directly off the env in _setup_contraction — so this is a no-op
    for that path. A no-op (angle_idx=[]) here is also harmless: every
    consumer treats an empty angle_idx as "nothing to embed".

    ``pos_dim`` is the width of the leading translation-invariant state block
    (see angle_utils' translation quotient). It travels with ``angle_idx``
    because every consumer needs both to size and build its input; 0 keeps the
    previous absolute-observation behaviour.
    """
    if not angle_idx and sym is None:
        return
    for block in agent_cfg.get("models", {}).values():
        if isinstance(block, dict):
            if angle_idx:
                block.setdefault("angle_idx", angle_idx)
            if sym is not None:
                block.setdefault("sym", sym)




def disable_tensorboard_files() -> None:
    """Stop skrl's SummaryWriter from touching disk, without breaking wandb logging.

    skrl's ``SummaryWriter.__init__`` (skrl.utils.tensorboard) eagerly
    constructs a real ``EventFileWriter``, which creates the run's tensorboard
    event file on disk the moment the agent is built — regardless of whether
    ``add_scalar`` is ever called. Those event files were the other big
    contributor to the cluster's inode quota (alongside wandb/, see
    ``install_wandb_scalar_hook``). ``install_wandb_scalar_hook`` mirrors
    every ``add_scalar`` call into wandb by wrapping this same class, so
    wandb logging must keep working after this call — only the on-disk
    writer is replaced with a no-op stub.
    """
    import skrl.utils.tensorboard as _skrl_tb

    class _NoopEventFileWriter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_event(self, event) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    _skrl_tb.EventFileWriter = _NoopEventFileWriter


def install_wandb_scalar_hook() -> None:
    """Mirror every skrl ``SummaryWriter.add_scalar`` into the active W&B run.

    Logged against a custom step metric (``global_step``) rather than wandb's
    internal step counter: any ``wandb.log()`` without ``step=`` (video/media
    uploads, the PathTracking figures) advances that internal counter, after
    which scalars logged with an explicit smaller ``step=`` are silently dropped
    ("steps must be monotonically increasing"). A step *metric* has no
    monotonicity requirement.

    Process-wide and shared by both routes, which is also what lets C2RL's inner
    PPO/SAC sub-agents reach the same run without ever calling ``wandb.init()``
    themselves (see ``rl_glue.make_base_rl_cfg``).
    """
    import skrl.utils.tensorboard as _skrl_tb
    import wandb

    orig_add_scalar = _skrl_tb.SummaryWriter.add_scalar
    defined = [False]

    def _add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        orig_add_scalar(self, tag=tag, value=value, timestep=timestep)
        if wandb.run is None:
            return
        if not defined[0]:
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            defined[0] = True
        wandb.log({tag: value, "global_step": timestep})

    _skrl_tb.SummaryWriter.add_scalar = _add_scalar


def apply_wandb_sweep_overrides(agent_cfg: dict) -> None:
    """Fold ``wandb.config`` (dotted keys) into ``agent_cfg`` for a sweep run.

    A sweep parameter whose value is itself a dict (wandb's nested-parameter
    form for jointly-sampled pairs, e.g. ``{w_lb, w_ub}`` — see
    ``search/configs/``) is merged into the existing sub-dict rather than
    assigned: assigning would wipe sibling keys already at that path
    (``agent.class``, ``agent.lbd``, …) that the sweep isn't sampling.

    Two synthetic, non-dotted keys fan a single sampled value out to both the
    actor's and the critic's reference-trajectory encoder, so one bayes trial
    always uses the same encoder (and stride) on both sides rather than
    sampling them independently — wandb's nested-dict form only joins keys
    that share a parent path (e.g. ``cm.w_lb``/``cm.w_ub``), which
    ``models.policy.encoder``/``models.critic.encoder`` don't:
      ``xref_encoder``        -> models.policy.encoder and models.critic.encoder
      ``xref_encoder_stride`` -> models.policy.encoder_stride and
                                  models.critic.encoder_stride
    See search/configs/c2rl-ppo-cvstem.yaml.
    """
    import wandb

    _FANOUT = {
        "xref_encoder": (("models", "policy", "encoder"),
                          ("models", "critic", "encoder")),
        "xref_encoder_stride": (("models", "policy", "encoder_stride"),
                                 ("models", "critic", "encoder_stride")),
    }

    for dotted, value in wandb.config.items():
        if dotted in _FANOUT:
            for *parents, leaf in _FANOUT[dotted]:
                node = agent_cfg
                for key in parents:
                    node = node.setdefault(key, {})
                node[leaf] = value
            continue
        *parents, leaf = dotted.split(".")
        node = agent_cfg
        for key in parents:
            node = node.setdefault(key, {})
        if isinstance(value, dict) and isinstance(node.get(leaf), dict):
            node[leaf].update(value)
        else:
            node[leaf] = value


def _disarm_termination_for_eval(env, tag: str = "[Eval]") -> None:
    """Force the early-termination box off on an evaluation env.

    Evaluation measures AUC = integral of ||e||/||e0|| over the fixed horizon.
    Cutting an episode short does not just shorten that integral, it inverts the
    metric: a policy that falls at step 20 stops accumulating error and scores a
    smaller (better-looking) AUC than one that tracks imperfectly for all 500
    steps. The same truncation makes the number incomparable with every
    LQR/C3M/CV-STEM baseline, all measured over full episodes.

    Early termination is a training-data intervention -- it keeps 500 steps of
    already-fallen transitions out of the rollout batch. It is not a measurement
    change, so it is disarmed here regardless of the training setting.
    """
    unwrapped = getattr(env, "unwrapped", env)
    if getattr(unwrapped, "terminate_out_of_box", False):
        unwrapped.set_terminate_out_of_box(False)
        print(f"{tag} early termination disarmed for evaluation — AUC is defined on the "
              f"full horizon, and truncating it rewards a policy for failing sooner.")


def apply_cli_dotted_overrides(agent_cfg: dict, extra_args: list[str]) -> list[str]:
    """Fold CLI ``--a.b.c=value`` args into ``agent_cfg``; return what was applied.

    Same dotted-path semantics as :func:`apply_wandb_sweep_overrides`, so a knob
    can be pinned by hand exactly the way a sweep would set it
    (``--agent.mini_batches=16``). Without this these args land in argparse's
    leftovers and are silently dropped on the classic route — the run then trains
    on the yaml defaults while its command line claims otherwise, which is the
    same class of silent-config failure as an unknown yaml key.

    The leaf must already exist in the yaml. A typo would otherwise create a dead
    key that ``rl_glue.filter_cfg_fields`` drops later without a word — the exact
    failure this function exists to prevent.
    """
    import ast

    applied = []
    for arg in extra_args:
        if not (arg.startswith("--") and "=" in arg and "." in arg.split("=", 1)[0]):
            continue
        dotted, raw = arg[2:].split("=", 1)
        try:
            # Literal first ("16" -> int, "0.3" -> float, "true"/"mlp" fall
            # through to str), so a numeric knob never arrives as a string.
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = {"true": True, "false": False, "null": None}.get(raw.lower(), raw)

        *parents, leaf = dotted.split(".")
        node = agent_cfg
        for key in parents:
            if not isinstance(node.get(key), dict):
                raise SystemExit(
                    f"[train] --{dotted}: no '{key}' section in this env's config — "
                    "the override would be silently dropped.")
            node = node[key]
        if leaf not in node:
            raise SystemExit(
                f"[train] --{dotted}: '{leaf}' is not a key of "
                f"{'.'.join(parents) or 'the config'} (have: "
                f"{', '.join(sorted(node)[:12])}...). Refusing to set a key the "
                "config does not declare — it would be dropped without a word.")
        node[leaf] = value
        applied.append(f"{dotted}={value!r}")
    if applied:
        print("[train] CLI overrides: " + "  ".join(applied))
    return applied


def finish_wandb(args_cli) -> None:
    """Close the active W&B run, if this process started one."""
    if getattr(args_cli, "no_wandb", False):
        return
    wandb = sys.modules.get("wandb")
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def normalize_agent_cfg(agent_cfg: dict, *, algorithm: str) -> dict:
    """Strip this repo's non-skrl ``agent:`` keys and resolve the preprocessors.

    Shared by both of train.py's routes so a yaml key means the same thing on
    each. Returns the std-dev-annealing decision, which the caller passes to
    ``agent_patches.patch_ppo_std_annealing`` after the agent is built.

    Observation/state normalization is disabled unconditionally: the Mahalanobis
    reward and the CV-STEM metric are defined in raw physical coordinates, and
    per-dimension scaling would distort the tracking error ``e = x - xref`` (see
    c2rl.py's module docstring). Value normalization stays available for PPO,
    where it only rescales the critic target.
    """
    from c2rl.agents.skrl.runner import CONTROL_BACKBONES

    a = agent_cfg.setdefault("agent", {})
    a.pop("use_state_norm", None)
    use_value_norm = a.pop("use_value_norm", True)
    # C2RL builds its own PPO/SAC sub-agent from the raw yaml dict via
    # rl_glue.make_base_rl_cfg, which reads ``use_value_norm`` itself (defaulting
    # to True when absent). Popping it here therefore made the key invisible to
    # C2RL: the `algorithm == "ppo"` branch below never fires for "c2rl-ppo", so
    # the flag was consumed, discarded, and then silently re-defaulted to True —
    # i.e. `use_value_norm: false` / `--use_value_norm false` did nothing at all
    # on any c2rl run, and the critic always got a RunningStandardScaler.
    # Put it back for the contraction algorithms, which are the ones that read it
    # downstream. (Plain ppo/sac still consume it here, as before.)
    if algorithm not in ("ppo", "sac"):
        a["use_value_norm"] = use_value_norm
    for key in ("state_preprocessor", "state_preprocessor_kwargs",
                "observation_preprocessor", "observation_preprocessor_kwargs"):
        a.pop(key, None)

    if use_value_norm and algorithm == "ppo":
        a["value_preprocessor"] = "RunningStandardScaler"
        a["value_preprocessor_kwargs"] = None
    else:
        a.pop("value_preprocessor", None)
        a.pop("value_preprocessor_kwargs", None)

    # Legacy annealing spellings — superseded by the backbone-driven decision
    # below, but still present in older configs, and skrl's PPO_CFG/SAC_CFG
    # would reject them as unknown fields.
    for key in ("anneal_stddev", "anneal_log_std"):
        a.pop(key, None)
    # std_dev_annealing/_kwargs follow the reward_euclidean rule below: skrl's
    # PPO_CFG/SAC_CFG reject them, but C2RL builds its sub-agent from this same
    # dict via rl_glue.make_base_rl_cfg and reads both off C2RLPPOCfg. Popping
    # unconditionally made them invisible to C2RL, which then fell back to the
    # dataclass/patch defaults: `std_dev_annealing: true` (so
    # `--std_dev_annealing false` did nothing) and kwargs=None (so the yaml
    # schedule was ignored and log_std annealed linearly to -2.0, sigma~=0.135,
    # instead of exponentially to the configured final_log_std).
    # reward_euclidean/reward_level are env-side switches (applied straight to
    # the raw env in train.py's standalone branch — see _apply_agent_overrides'
    # caller), not real PPO_CFG/SAC_CFG fields. C2RL reads them off its own
    # dataclass instead (c2rl.py), so only pop for the two skrl-native algos —
    # left in place they'd raise the same "unexpected keyword argument" crash
    # the `models` dotted-path bug did.
    std_kwargs = a.get("std_dev_annealing_kwargs")
    if algorithm in ("ppo", "sac"):
        a.pop("reward_euclidean", None)
        a.pop("reward_level", None)
        a.pop("std_dev_annealing", None)
        a.pop("std_dev_annealing_kwargs", None)
    return {
        "std_dev_annealing": (
            agent_cfg.get("models", {}).get("policy", {}).get("backbone") in CONTROL_BACKBONES
        ),
        "std_dev_annealing_kwargs": std_kwargs,
    }


def apply_agent_patches(agent, *, algorithm: str, annealing: dict,
                        namespace: bool = True) -> None:
    """Apply every post-construction patch an agent needs, in dependency order.

    Each patch no-ops when the agent lacks the attribute it hooks, so this is
    safe to call on any agent — including C2RL's outer agent, which owns no
    ``.policy``/``.scheduler``/``.scaler`` and patches its inner PPO/SAC
    sub-agent itself (see c2rl.py).

    ``patch_auc_checkpoint`` goes last because it wraps ``post_interaction``,
    and it must see (and therefore run after) the annealing wrapper installed by
    ``patch_ppo_std_annealing``. ``namespace=False`` for the contraction
    algorithms, which already namespace their own ``track_data`` keys.
    """
    from c2rl.agents.skrl.agent_patches import (
        best_metric_for,
        patch_algo_namespace,
        patch_auc_checkpoint,
        patch_kl_logging,
        patch_ppo_std_annealing,
        patch_prune_checkpoints,
        patch_sac_entropy_clamp,
    )

    patch_kl_logging(agent)
    patch_sac_entropy_clamp(agent)
    patch_ppo_std_annealing(agent, annealing["std_dev_annealing"],
                            annealing["std_dev_annealing_kwargs"])
    if namespace:
        patch_algo_namespace(agent, algorithm.upper())
    # Wraps write_checkpoint (not post_interaction), so it is independent of
    # the patch_auc_checkpoint ordering rule above.
    patch_prune_checkpoints(agent)
    patch_auc_checkpoint(agent, metric=best_metric_for(algorithm))


def _max_step_reward(robot: str, env_cfg) -> float:
    """Best-case per-step reward for a vel-tracking env's reward function.

    Sum of the maxima of every reward term that can be positive; terms of the
    form `nonneg_quantity * non_positive_scale` (all the tracking/regularization
    penalties) have a best case of 0 and are omitted.

    quadruped/humanoid: alive bonus + the two exp-tracking terms (each saturates
    at its scale when the tracking error is 0). The quadruped also has a gait
    term `(2*gait_score - 1) * rew_gait`, gait_score in [0, 1], whose best case
    is `+rew_gait` (>0) — it must be included or the "theoretical max" it feeds
    (0.5 * max * T for the ref-traj quality gate) is under-counted, so an
    actually-achievable return can exceed it (observed: a real run hit ~6200
    against a mis-computed 5200 ceiling). Humanoid has no gait term. manipulator
    has no alive bonus and every term is `error * negative_scale`, best case 0.
    """
    if robot in ("quadruped", "humanoid"):
        # rew_gait absent on humanoid (getattr default 0.0); its max contribution
        # is +rew_gait (gait_score = 1 → (2*1 - 1)*rew_gait).
        return (env_cfg.rew_alive + env_cfg.rew_lin_vel + env_cfg.rew_yaw_rate
                + max(0.0, getattr(env_cfg, "rew_gait", 0.0)))
    if robot == "manipulator":
        return 0.0
    raise ValueError(f"no max-reward formula for robot '{robot}'")


def _generate_ref_trajs(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli):
    import numpy as np
    import torch

    robot = next((name for prefix, name in _VEL_TASK_TO_ROBOT.items() if task.startswith(prefix)), None)
    if robot is None:
        print(f"[RefTraj] No robot mapping for task '{task}'; skipping.")
        return

    out_dir = os.path.join(_DATA_ROOT, robot)
    out_path = os.path.join(out_dir, "dynamics_data.npz")
    T = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    # Quality threshold = half of the theoretical best-case total episode reward
    # (best-case per-step reward × T), rather than a hand-picked constant — this
    # tracks whatever the reward scales in env_cfg actually are per task.
    min_reward = args_cli.min_ref_quality if args_cli.min_ref_quality is not None \
        else 0.5 * _max_step_reward(robot, env_cfg) * T

    # Load best checkpoint
    import logging as _logging
    _skrl_log = _logging.getLogger("skrl")
    _prev_level = _skrl_log.level
    _skrl_log.setLevel(_logging.ERROR)
    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best_ckpt):
        print(f"[RefTraj] Loading best checkpoint: {best_ckpt}")
        agent.load(best_ckpt)
    else:
        print("[RefTraj] WARNING: best_agent.pt not found; using final weights.")
    _skrl_log.setLevel(_prev_level)
    for model in agent.models.values():
        if model is not None:
            model.eval()

    def _get_obs(o):
        return o["policy"] if isinstance(o, dict) else o

    unwrapped = isaac_env.unwrapped
    _act_low = torch.as_tensor(skrl_env.action_space.low, dtype=torch.float32, device=skrl_env.device)
    _act_high = torch.as_tensor(skrl_env.action_space.high, dtype=torch.float32, device=skrl_env.device)

    # Quality gate: 1 full episode across all parallel environments
    if min_reward > 0:
        print(f"\n[RefTraj] Evaluating quality (threshold: mean total reward >= {min_reward}) …")

        # Deliberately measures the same quantity as training's "Reward / Total
        # reward (mean)": min_reward is calibrated as 0.5 * best-case-per-step *
        # T, i.e. on a full training episode's scale. So this rollout must
        # reproduce training's episode structure — fall termination at its cfg
        # default (True) and the policy's own stochastic actions.
        #
        # It must not disable terminate_on_fall the way _evaluate_best_model
        # does: with it off a fallen robot flails for the full T steps,
        # accumulating the lying-down penalties at ~-2.5/step — a large negative
        # tail training never sees, since it resets on fall. That measurement
        # produced gate values of -545 against a +2600 threshold, structurally
        # unreachable at any policy quality, while the same policy's training
        # reward was well positive.
        ep_rewards = []
        ep_r = torch.zeros(skrl_env.num_envs, device=skrl_env.device)
        finished = torch.zeros(skrl_env.num_envs, dtype=torch.bool, device=skrl_env.device)

        if hasattr(skrl_env, "_reset_once"):
            skrl_env._reset_once = True
        obs_dict, _ = skrl_env.reset()
        obs = _get_obs(obs_dict)

        # We run for slightly more than T steps to ensure all envs finish their first episode
        for _ in range(T + 1):
            with torch.no_grad():
                actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
            obs_dict, rewards, terminated, truncated, _ = skrl_env.step(actions)
            obs = _get_obs(obs_dict)

            # accumulate reward only for envs that haven't finished their first episode
            ep_r += rewards.squeeze(-1) * (~finished).float()
            done = (terminated | truncated).squeeze(-1)

            # only record the reward when an env finishes its first episode
            just_finished = done & (~finished)
            for i in just_finished.nonzero(as_tuple=True)[0]:
                ep_rewards.append(ep_r[i.item()].item())

            finished |= done
            if finished.all():
                break

        # If any envs somehow didn't finish, we record their accumulated rewards
        not_finished = ~finished
        for i in not_finished.nonzero(as_tuple=True)[0]:
            ep_rewards.append(ep_r[i.item()].item())

        if not ep_rewards:
            print("[RefTraj] WARNING: no complete episodes; skipping.")
            return
        mean_r = sum(ep_rewards) / len(ep_rewards)
        print(f"[RefTraj] Mean total reward: {mean_r:.1f}")
        if mean_r < min_reward:
            print(
                f"[RefTraj] SKIPPED — policy quality too low "
                f"({mean_r:.1f} < {min_reward}). Train longer or pass --min_ref_quality 0."
            )
            return

    # Collect trajectories. We over-collect a candidate pool larger than
    # num_trajs (oversample_factor x) and then keep the longest num_trajs of
    # them — early termination is exactly what a poor/failing rollout looks
    # like, so ranking by survival length is a simple, direct proxy for
    # "better trajectory". Recording every one of num_envs (rather than just
    # the first min(num_trajs, num_envs)) maximizes that pool for free: Isaac
    # can't shrink the batch, so the extra envs are being simulated regardless.
    # This also means num_envs < num_trajs naturally loops through as many
    # per-env episode rounds as it takes to fill the pool — no special-casing
    # needed for that direction.
    import math

    import tqdm

    num_trajs = args_cli.ref_num_trajs
    pool_target = max(num_trajs, int(math.ceil(num_trajs * max(1.0, args_cli.ref_oversample_factor))))
    print(f"[RefTraj] Collecting a candidate pool of {pool_target} trajectories "
          f"(oversample x{args_cli.ref_oversample_factor:g}), keeping the longest {num_trajs} → {out_path}")
    num_envs = skrl_env.num_envs
    all_states, all_actions, all_lengths = [], [], []
    if hasattr(skrl_env, "_reset_once"):
        skrl_env._reset_once = True
    obs_dict, _ = skrl_env.reset()
    obs = _get_obs(obs_dict)

    # _act_low/_act_high (defined above, alongside `unwrapped`) are used only
    # when writing into the saved `u` array below, never to modify what's
    # stepped through the env. The policy samples with clip_actions=False
    # (clipping inside the actor corrupts the log-prob), and the env already
    # enforces action bounds on its own (its actuator/physics pipeline), so
    # re-clipping before `skrl_env.step()` would be redundant. But the *saved*
    # dynamics_data.npz must record actions within the declared action space —
    # an unclipped, possibly out-of-range sample is not a valid "u" for
    # fitting f(x) + B(x)u.

    # Pre-allocate tensors to avoid massive python list overhead
    with torch.no_grad():
        actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
    state_tensor = unwrapped.get_physical_state()
    state_dim = state_tensor.shape[1]
    u_dim = actions.shape[1]

    ep_states = torch.zeros((num_envs, T, state_dim), dtype=torch.float32, device=skrl_env.device)
    ep_actions = torch.zeros((num_envs, T, u_dim), dtype=torch.float32, device=skrl_env.device)
    step_counts = torch.zeros(num_envs, dtype=torch.long, device=skrl_env.device)

    pbar = tqdm.tqdm(total=pool_target, desc="[RefTraj] Collecting candidates")

    while len(all_states) < pool_target:
        with torch.no_grad():
            actions, _ = agent.act(obs, None, timestep=0, timesteps=0)
        state_tensor = unwrapped.get_physical_state()

        # Record state and action for envs that are still within T steps
        valid_mask = step_counts < T
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        ep_states[valid_indices, step_counts[valid_indices]] = state_tensor[valid_indices].float()
        # Clip only for the saved record, not for stepping (see note above).
        ep_actions[valid_indices, step_counts[valid_indices]] = \
            torch.clamp(actions[valid_indices], _act_low, _act_high).float()

        step_counts[valid_indices] += 1

        obs_dict, _, terminated, truncated, _ = skrl_env.step(actions)
        obs = _get_obs(obs_dict)
        done = (terminated | truncated).squeeze(-1)

        if done.any():
            done_indices = done.nonzero(as_tuple=True)[0]
            # Accept trajectories that survived at least min_ref_traj_length_frac
            # of the max length (default 0.5 = half of T). This handles policies
            # that fall slightly early but pass the quality gate, as well as
            # off-by-one errors with Isaac Gym's max_episode_length.
            min_len = int(args_cli.min_ref_traj_length_frac * T)
            success_mask = step_counts[done_indices] >= min_len
            success_indices = done_indices[success_mask]

            if len(success_indices) > 0:
                # Pad any missing steps with the final valid state to ensure x_dot is stable
                for i in success_indices:
                    length = step_counts[i].item()
                    if length < T and length > 0:
                        ep_states[i, length:] = ep_states[i, length - 1].clone()
                        ep_actions[i, length:] = ep_actions[i, length - 1].clone()

                # Move to CPU in bounded chunks. Episodes are length-synchronized
                # (fixed T), so on the first `done` event success_indices can be
                # ~num_envs at once — gathering all of them in one fancy-index
                # would allocate a full (len(success_indices), T, dim) CUDA
                # temporary, so keep it chunked regardless of pool size.
                #
                # Deliberately don't early-break once len(all_states) hits
                # pool_target here: this round's successes are already sitting
                # in GPU memory finished at the same time, so cutting the chunk
                # loop short would arbitrarily favor low env-index trajectories
                # over otherwise-equal ones later in `success_indices`. Letting
                # the whole round in (pool may overshoot pool_target a bit)
                # keeps every env that finished this round in the running for
                # the final longest-num_trajs selection.
                _CHUNK = 256
                for start in range(0, len(success_indices), _CHUNK):
                    idx = success_indices[start:start + _CHUNK]
                    s_np = ep_states[idx].cpu().numpy()
                    a_np = ep_actions[idx].cpu().numpy()
                    l_np = step_counts[idx].cpu().numpy()
                    for i in range(len(idx)):
                        all_states.append(s_np[i])
                        all_actions.append(a_np[i])
                        all_lengths.append(int(l_np[i]))
                        pbar.update(1)

            # Reset the step counts for all finished environments
            step_counts[done_indices] = 0

    pbar.close()

    # Keep the num_trajs longest candidates out of the oversampled pool.
    all_lengths_np = np.asarray(all_lengths, dtype=np.int64)
    keep = np.argsort(all_lengths_np)[::-1][:num_trajs]
    print(f"[RefTraj] Pool lengths: min={all_lengths_np.min()}, max={all_lengths_np.max()}, "
          f"median={int(np.median(all_lengths_np))} (T={T}) — keeping top {num_trajs} by length")
    states_arr = np.stack([all_states[i] for i in keep]).astype(np.float32)
    actions_arr = np.stack([all_actions[i] for i in keep]).astype(np.float32)
    lengths_arr = all_lengths_np[keep]

    os.makedirs(out_dir, exist_ok=True)

    # The diagnostic plot is a log, not data — it documents this generation run
    # rather than feeding a later one, so it goes to logs/ and leaves data/
    # holding only the npz artifacts other algorithms consume.
    plot_dir = os.path.join(_ROOT, "logs", robot)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot absolute position of 10 sampled trajectories
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for i in range(min(10, num_trajs)):
            plt.plot(states_arr[i, :, 0], states_arr[i, :, 1], label=f"Traj {i+1}")
        plt.xlabel("X Position (m, relative)")
        plt.ylabel("Y Position (m, relative)")
        plt.title("Sampled Reference Trajectories (Relative Position)")
        plt.legend()
        plot_path = os.path.join(plot_dir, "position_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[RefTraj] Saved position plot → {plot_path}")
    except Exception as e:
        print(f"[RefTraj] Failed to generate position plot: {e}")

    # Generate dynamics data via finite differences
    dt = env_cfg.sim.dt * env_cfg.decimation
    print(f"[RefTraj] Computing dynamics (x_dot) via 4th-order central difference (dt={dt:.3f})...")

    # angle_idx columns (e.g. yaw) wrap at +-pi in the saved states_arr — a raw
    # finite difference across that wrap would spike x_dot by ~2*pi/dt for one
    # sample. Difference an unwrapped copy instead (np.unwrap makes each angle
    # column continuous by adding +-2*pi at jumps); states_arr itself (saved as
    # `x` below) is left untouched — NeuralDynamics only ever consumes x through
    # its (cos, sin) embedding, which is identical for theta and theta + 2*pi*k,
    # so this is purely a finite-difference cleanup, not a semantic change to x.
    angle_idx = list(getattr(isaac_env.unwrapped, "angle_idx", []) or [])
    diff_states = states_arr
    if angle_idx:
        diff_states = states_arr.copy()
        for idx in angle_idx:
            diff_states[:, :, idx] = np.unwrap(diff_states[:, :, idx], axis=1)

    x_dot_arr = np.zeros_like(states_arr)
    for i in range(2, diff_states.shape[1] - 2):
        x_dot_arr[:, i] = (-diff_states[:, i + 2] + 8 * diff_states[:, i + 1] - 8 * diff_states[:, i - 1] + diff_states[:, i - 2]) / (12 * dt)
    # Forward/backward differences for boundaries
    x_dot_arr[:, 0] = (-3 * diff_states[:, 0] + 4 * diff_states[:, 1] - diff_states[:, 2]) / (2 * dt)
    x_dot_arr[:, 1] = (-3 * diff_states[:, 1] + 4 * diff_states[:, 2] - diff_states[:, 3]) / (2 * dt)
    x_dot_arr[:, -2] = (3 * diff_states[:, -2] - 4 * diff_states[:, -3] + diff_states[:, -4]) / (2 * dt)
    x_dot_arr[:, -1] = (3 * diff_states[:, -1] - 4 * diff_states[:, -2] + diff_states[:, -3]) / (2 * dt)

    # Filter out any episodes that contain NaNs
    nan_mask = np.isnan(states_arr).any(axis=(1, 2)) | np.isnan(actions_arr).any(axis=(1, 2)) | np.isnan(x_dot_arr).any(axis=(1, 2))
    if nan_mask.any():
        num_nans = nan_mask.sum()
        print(f"[RefTraj] WARNING: Found NaNs in {num_nans} episodes! Filtering them out before saving...")
        valid_mask = ~nan_mask
        states_arr = states_arr[valid_mask]
        actions_arr = actions_arr[valid_mask]
        x_dot_arr = x_dot_arr[valid_mask]
        lengths_arr = lengths_arr[valid_mask]

    # Single unified file: reference trajectories are the (x, u) part of the
    # dynamics data, so there is no separate ref_trajs.npz anymore.
    #   x       (N, T, x_dim)  physical states; steps >= lengths[n] are padding
    #                          (the last valid state repeated, keeping x_dot ~ 0)
    #   u       (N, T, u_dim)  executed (clipped) actions, same padding rule
    #   x_dot   (N, T, x_dim)  4th-order central differences of x
    #   lengths (N,)           number of valid steps per trajectory — consumers
    #                          mask with arange(T) < lengths[:, None]
    dyn_path = os.path.join(out_dir, "dynamics_data.npz")
    np.savez_compressed(dyn_path, x=states_arr, u=actions_arr, x_dot=x_dot_arr, lengths=lengths_arr)
    print(f"[RefTraj] Saved dynamics  → {dyn_path}")
    print(f"       x       shape: {states_arr.shape}")
    print(f"       u       shape: {actions_arr.shape}   (clipped to action-space bounds)")
    print(f"       x_dot   shape: {x_dot_arr.shape}")
    print(f"       lengths shape: {lengths_arr.shape}  (min {lengths_arr.min()}, max {lengths_arr.max()})")


def _evaluate_classic_path_tracking(*, task, runner, args_cli, _is_classic, num_groups: int = 10,
                                    episodes_per_group: int = 5, label: str = "",
                                    held_out_seed: int | None = None, held_out_trajectories: int = 64):
    """Post-training evaluation for classic path-tracking envs (CAC-dev style).

    Classic envs are plain non-vectorized gymnasium Envs with variable-length
    episodes and no early termination (only truncation at the sampled length),
    so unlike the Isaac rollout there is no terminate_on_fall concept and no
    vectorized-boundary bookkeeping: a plain loop over one env instance, one
    episode at a time, using its native ``tracking_error``/``dt`` step info.

    Reports mean +/- 95% CI of total reward, error AUC, and overshoot C /
    contraction rate lambda from the minimal-AUC envelope C*exp(-lambda*k*dt).

    ``held_out_seed`` (UVFA-style generalization test): evaluates on a fixed
    bank of ``held_out_trajectories`` shapes from a generator seeded
    independently of training — guaranteed unseen, unlike this function's
    ordinary i.i.d. eval trajectories (a fresh draw from the same distribution,
    not a held-out slice of it). Pair with a second call using
    ``label="HeldOut"`` so both land in separate ``eval_results.json``.

    Skipped by ``--skip_final_eval``. Sequential over one env instance (50
    episodes), fine as an end-of-training report but dead weight in a sweep: it
    does not feed the sweep metric at all (that ``Stability/auc_mean`` comes
    from StatManagerEnvWrapper during the trainer loop; the ``auc_mean`` here is
    a separate, differently-scoped number). For an SDP-per-step controller like
    online CV-STEM-LQR it is also the most expensive thing in the trial.
    """
    if getattr(args_cli, "skip_final_eval", False):
        print("[Eval] SKIPPED — --skip_final_eval (does not feed the sweep metric).")
        return

    import json

    import gymnasium as gym
    import numpy as np
    import torch
    from c2rl.tasks.direct.common.eval_metrics import (
        fit_exponential_envelope,
        mean_confidence_interval,
    )

    probe = gym.make(task)
    if not hasattr(probe.unwrapped, "xref"):
        print(f"[Eval] SKIPPED — env {type(probe.unwrapped).__name__} has no reference trajectory (xref).")
        probe.close()
        return
    probe.close()

    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best_ckpt):
        print(f"[Eval] Loading best checkpoint: {best_ckpt}")
        agent.load(best_ckpt)
    else:
        print("[Eval] WARNING: best_agent.pt not found; evaluating final weights.")
    for model in agent.models.values():
        if model is not None:
            model.eval()

    device = agent.device
    env = gym.make(task, device=device)
    _disarm_termination_for_eval(env, "[Eval]")

    if held_out_seed is not None:
        if hasattr(env.unwrapped, "set_held_out_mode"):
            env.unwrapped.set_held_out_mode(held_out_seed, held_out_trajectories)
            print(f"[Eval] held-out mode: {held_out_trajectories} FIXED reference-trajectory "
                  f"shapes (generator seed={held_out_seed}, disjoint from training's RNG stream).")
        else:
            print(f"[Eval] --eval_held_out_seed set but {type(env.unwrapped).__name__} has no "
                  f"set_held_out_mode — falling back to ordinary i.i.d.-random eval trajectories.")

    # Mirror the training env's reference window so the eval observation matches
    # what the models were built for. A mismatch is not a shape error that
    # surfaces loudly — RefWindow.split would raise, but only after the eval env
    # has already been built — so copy the layout explicitly.
    from c2rl.runners.contraction_runner import _unwrap_env
    _train_env = _unwrap_env(getattr(runner, "_env", None)) if getattr(runner, "_env", None) is not None else None
    _w = getattr(_train_env, "ref_window", None)
    if _w is not None and hasattr(env.unwrapped, "configure_ref_window"):
        env.unwrapped.configure_ref_window(length=_w.length, offset=_w.offset)
        print(f"[Eval] reference window mirrored: length={_w.length} offset={_w.offset}")

    reward_list, auc_list, C_list, lbd_list = [], [], [], []
    print(f"[Eval] Rolling out {num_groups * episodes_per_group} episodes on {task} …")
    for _g in range(num_groups):
        error_trajs = []
        for _e in range(episodes_per_group):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            error_traj = []
            dt = env.unwrapped.dt
            while not done:
                if isinstance(obs, dict):
                    # {x, xrefs, urefs} — flatten exactly as skrl does so the
                    # models see the same layout they were built for.
                    obs_t = env.unwrapped.ref_window.flatten(
                        *(torch.as_tensor(obs[k], dtype=torch.float32, device=device)
                          for k in ("x", "xrefs", "urefs")))
                elif isinstance(obs, torch.Tensor):
                    obs_t = obs.clone().detach().to(dtype=torch.float32, device=device)
                else:
                    obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device)
                if obs_t.dim() == 1:
                    obs_t = obs_t.unsqueeze(0)
                with torch.no_grad():
                    # see _evaluate_best_model for why agent.act() (not
                    # agent.policy.act()) is the algorithm-agnostic interface
                    actions, outputs = agent.act(obs_t, None, timestep=0, timesteps=0)
                    action = outputs.get("mean_actions", actions)
                obs, reward, terminated, truncated, info = env.step(action)

                term_val = terminated.item() if isinstance(terminated, torch.Tensor) else bool(terminated)
                trunc_val = truncated.item() if isinstance(truncated, torch.Tensor) else bool(truncated)
                done = term_val or trunc_val
                ep_reward += float(reward.item() if isinstance(reward, torch.Tensor) else reward)

                err_val = info["tracking_error"].item() if isinstance(info["tracking_error"], torch.Tensor) else info["tracking_error"]
                error_traj.append(float(np.sqrt(max(err_val, 0.0))))
            e0 = max(error_traj[0], 1e-8) if error_traj else 1.0
            norm_traj = np.asarray(error_traj) / e0
            error_trajs.append(norm_traj)
            reward_list.append(ep_reward)
            auc_list.append(float(np.trapezoid(norm_traj, dx=dt)) if hasattr(np, "trapezoid")
                             else float(np.trapz(norm_traj, dx=dt)))
        # paper fit: one overshoot C* per group, one convergence rate per curve
        C, lbds = fit_exponential_envelope(error_trajs, dt)
        C_list.append(C)
        lbd_list.extend(float(x) for x in lbds)
    # Read before close() — used for the divergence threshold below.
    _time_bound = float(getattr(env.unwrapped, "time_bound", 0.0) or 0.0)
    env.close()

    rew_mean, rew_ci = mean_confidence_interval(reward_list)
    auc_mean, auc_ci = mean_confidence_interval(auc_list)
    C_mean, C_ci = mean_confidence_interval(C_list)
    lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
    results = {
        **_run_metadata(args_cli, task),
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "num_episodes": num_groups * episodes_per_group,
        "total_reward_mean": rew_mean, "total_reward_ci95": rew_ci,
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
        "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
        "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
        "num_fit_groups": num_groups,
        # Raw per-episode contraction rates (one per evaluated episode, before
        # the running/population mean above collapses them) — kept alongside
        # contraction_rate_mean so a caller can distinguish a single episode's
        # fitted lambda from the running mean lambda over the eval population.
        "contraction_rate_all": [round(float(x), 6) for x in lbd_list],
        "overshoot_all": [round(float(x), 6) for x in C_list],
        # Raw per-episode AUCs. auc_mean +/- CI alone cannot distinguish the two
        # very different ways a seed ends up with a bad number: a uniform shift
        # (every episode slightly worse — a genuinely weaker controller) versus a
        # heavy tail (most episodes fine, a handful diverging — a robustness
        # failure). Those call for opposite fixes, and across-seed AUC spread is
        # exactly the symptom under investigation, so the per-episode
        # distribution is kept alongside the robust summaries below.
        "auc_all": [round(float(x), 6) for x in auc_list],
        "auc_median": float(np.median(auc_list)),
        "auc_p90": float(np.percentile(auc_list, 90)),
        "auc_max": float(np.max(auc_list)),
        # Fraction of episodes whose normalized error never really contracts.
        # AUC = int(||e||/||e0||) dt over a time_bound-long episode, so an
        # episode that merely holds its initial error scores ~time_bound.
        #
        # The threshold was 0.1*time_bound (= 1.5 at time_bound 15) on the
        # assumption that a contracting episode scores ~1. It does not: the
        # best arms measured run 1.48-2.06 per episode, i.e. At or above that
        # cut, so the metric labelled the strongest controller in the study
        # ~50% "diverged". Measured distribution across 1095 archived runs:
        #     contracting   1.5 - 2.1
        #     marginal      2.2 - 6.0
        #     diverged     12.4 - 81.0
        # 1/3 of time_bound (= 5.0) sits in the empty band between marginal and
        # diverged, ~3x above the good arms and ~2.5x below the failures. This
        # is a robustness heuristic for spotting heavy tails, not a sharp
        # classifier -- read it next to auc_p90/auc_max, never alone.
        "auc_diverged_frac": (float(np.mean(np.asarray(auc_list) > (1.0 / 3.0) * _time_bound))
                              if _time_bound > 0 else None),
    }

    _tag = f"[Eval{'-' + label if label else ''}]"
    print(f"{_tag} ── Best-model evaluation (classic path-tracking){(' — ' + label) if label else ''} ──")
    print(f"{_tag} total reward     : {rew_mean:.2f} ± {rew_ci:.2f} (95% CI, n={len(reward_list)})")
    print(f"{_tag} error AUC        : {auc_mean:.4f} ± {auc_ci:.4f}")
    print(f"{_tag} overshoot C      : {C_mean:.3f} ± {C_ci:.3f}")
    print(f"{_tag} contraction rate : {lbd_mean:.4f} ± {lbd_ci:.4f}  (C·e^(−λkΔt), min AUC)")

    # wandb first, disk second. Training has already succeeded by this point and
    # the numbers above are the result; a filesystem problem must not be able to
    # lose them. This ordering used to be reversed, and the disk write threw.
    if not args_cli.no_wandb and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
        wandb_logs = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if "reward" in k:
                    wandb_logs[f"Reward/{k}"] = v
                elif any(s in k for s in ["auc", "overshoot", "contraction_rate", "contraction_score"]):
                    wandb_logs[f"Stability/{k}"] = v
                else:
                    wandb_logs[f"final_eval/{k}"] = v
        sys.modules["wandb"].log(wandb_logs)

    # makedirs, because agent.experiment_dir may not exist. skrl creates it
    # lazily on its first write there, and for a sweep trial train.py sets
    # checkpoint_interval = 0 (hundreds of throwaway trials must not each write
    # ten checkpoints) -- so nothing ever writes, the directory is never made,
    # and this open() raised FileNotFoundError on every sweep trial that got as
    # far as evaluating. Training had already finished at that point, so a
    # multi-hour trial was being thrown away at the last step.
    _json_name = f"eval_results{'_' + label.lower() if label else ''}.json"
    out_json = os.path.join(agent.experiment_dir, _json_name)
    try:
        os.makedirs(agent.experiment_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Saved → {out_json}")
    except OSError as exc:
        # Loud, but not fatal. The metrics are already in wandb above, and the
        # json is a convenience copy: a full quota or a read-only path is not a
        # reason to discard a finished run. Raising here is what turned a
        # successful trial into a crashed one.
        print(f"[Eval] WARNING: could not write {out_json}: {exc}. "
              f"Metrics were logged to wandb; continuing.", flush=True)


def _evaluate_best_model(*, task, runner, isaac_env, skrl_env, env_cfg, args_cli, num_groups: int = 10):
    """Post-training evaluation of the best checkpoint (CAC-dev style).

    Loads best_agent.pt, disables fall termination (episodes always run the
    full length so metrics are comparable across policies), rolls out one full
    episode in every parallel env with deterministic (mean) actions clipped to
    the action space, and reports mean +/- 95% CI of:

      * total reward
      * AUC of the velocity-tracking error (trapezoid, dt-weighted)
      * contraction rate lambda and overshoot C — the exponential envelope
        C * exp(-lambda * k * dt) bounding the normalized error curves with
        minimal envelope AUC (= C/lambda), fitted per env-group (CAC-dev
        trainer/evaluator.py compute_contraction_rate).

    Results are printed, logged to wandb (if active), and saved as
    eval_results.json next to the checkpoints.

    Skipped entirely by ``--skip_final_eval`` — see
    _evaluate_classic_path_tracking for why a sweep does not want this.
    """
    if getattr(args_cli, "skip_final_eval", False):
        print("[Eval] SKIPPED — --skip_final_eval (does not feed the sweep metric).")
        return

    import json

    import numpy as np
    import torch
    from c2rl.tasks.direct.common.eval_metrics import (
        fit_exponential_envelope,
        mean_confidence_interval,
    )

    agent = runner.agent
    best_ckpt = os.path.join(agent.experiment_dir, "checkpoints", "best_agent.pt")
    if os.path.exists(best_ckpt):
        print(f"[Eval] Loading best checkpoint: {best_ckpt}")
        agent.load(best_ckpt)
    else:
        print("[Eval] WARNING: best_agent.pt not found; evaluating final weights.")
    for model in agent.models.values():
        if model is not None:
            model.eval()

    unwrapped = isaac_env.unwrapped
    # Same rule as the classic evaluator, and it bites harder here: this loop
    # runs a fixed T steps without checking dones, so an env that terminated
    # early would be auto-reset underneath it and every later
    # get_tracking_error() would silently measure a different episode against
    # the old e(0). Inert today (Isaac _state_bounds is ±inf, so the box never
    # fires) — disarmed anyway, because the day a subclass declares finite
    # bounds this would corrupt the error curve without any error message.
    _disarm_termination_for_eval(isaac_env, "[Eval]")
    dt = env_cfg.sim.dt * env_cfg.decimation
    T = int(env_cfg.episode_length_s / dt)
    num_envs = skrl_env.num_envs

    if not hasattr(unwrapped, "get_tracking_error"):
        print(f"[Eval] SKIPPED — env {type(unwrapped).__name__} has no get_tracking_error().")
        return

    _act_low = torch.as_tensor(skrl_env.action_space.low, dtype=torch.float32, device=skrl_env.device)
    _act_high = torch.as_tensor(skrl_env.action_space.high, dtype=torch.float32, device=skrl_env.device)

    # Non-terminating evaluation: flip the cfg flag (read every step by
    # _get_dones) and restore afterwards.
    prev_flag = getattr(unwrapped.cfg, "terminate_on_fall", True)
    unwrapped.cfg.terminate_on_fall = False
    try:
        if hasattr(skrl_env, "_reset_once"):
            skrl_env._reset_once = True
        obs_dict, _ = skrl_env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        total_reward = torch.zeros(num_envs, device=skrl_env.device)
        errors = torch.zeros(num_envs, T + 1, device=skrl_env.device)
        errors[:, 0] = unwrapped.get_tracking_error()

        print(f"[Eval] Rolling out {num_envs} non-terminating episodes of {T} steps …")
        for k in range(T):
            with torch.no_grad():
                # agent.act() is the uniform interface across every skrl Agent
                # (PPO/SAC/C3M/C2RL/SDLQR/LQR) — unlike agent.policy.act(...),
                # which assumes PPO/SAC's internal attribute names and breaks
                # on contraction agents. "mean_actions" (present for Gaussian
                # policies) gives the deterministic action; deterministic
                # policies (e.g. C3M's CLDeterministicActorModel) have no
                # separate mean, so their raw action is already deterministic.
                actions, outputs = agent.act(obs, None, timestep=0, timesteps=0)
                actions = torch.clamp(outputs.get("mean_actions", actions), _act_low, _act_high)
            obs_dict, rewards, terminated, truncated, _ = skrl_env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            total_reward += rewards.squeeze(-1)
            errors[:, k + 1] = unwrapped.get_tracking_error()
    finally:
        unwrapped.cfg.terminate_on_fall = prev_flag

    err_np = errors.cpu().numpy()  # (N, T+1)
    rew_np = total_reward.cpu().numpy()

    # Cap the Stability-tab sample size to the SAC-family env count, regardless
    # of how many parallel envs this run actually used. PPO-family algorithms
    # train/roll out with far more parallel envs (e.g. 4096) than SAC-family
    # ones (64, see _DEFAULT_NUM_ENVS_SAC) — without this cap, PPO's mean/CI
    # would be computed from a much larger sample than SAC's, so the two
    # wouldn't be comparable on the Stability tab. Truncating (not resampling)
    # keeps this deterministic across reruns.
    if num_envs > _DEFAULT_NUM_ENVS_SAC:
        num_envs = _DEFAULT_NUM_ENVS_SAC
        err_np = err_np[:num_envs]
        rew_np = rew_np[:num_envs]

    # AUC over the normalized error curve (dt-weighted trapezoid), per episode.
    # np.trapezoid is numpy>=2 only; env_isaaclab ships numpy 1.26 (trapz).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    e0_np = err_np[:, 0]
    e0_np_safe = np.maximum(e0_np, 1e-8)
    norm_err_np = err_np / e0_np_safe[:, None]
    auc_np = _trapz(norm_err_np, dx=dt, axis=1)

    # Contraction envelope on normalized error e(t)/e(0) — CAC-dev convention.
    # Envs whose initial error is ~0 (near-zero commanded velocity) carry no
    # contraction information and are excluded from the fit.
    e0 = err_np[:, 0]
    fit_mask = e0 > 0.05
    C_list, lbd_list, score_list = [], [], []
    fit_ids = np.nonzero(fit_mask)[0]
    if len(fit_ids) >= num_groups:
        groups = np.array_split(fit_ids, num_groups)
        for g in groups:
            # raw error curves; fit_exponential_envelope normalizes by e(0) itself
            raw_trajs = [err_np[i] for i in g]
            C, lbds = fit_exponential_envelope(raw_trajs, dt)
            C_list.append(C)
            lbd_list.extend(float(x) for x in lbds)
            score_list.extend(float(x) / max(C, 1e-6) for x in lbds)
    else:
        print(f"[Eval] WARNING: only {len(fit_ids)} envs with e(0) > 0.05; skipping contraction fit.")

    rew_mean, rew_ci = mean_confidence_interval(rew_np)
    auc_mean, auc_ci = mean_confidence_interval(auc_np)
    results = {
        **_run_metadata(args_cli, task),
        "checkpoint": best_ckpt if os.path.exists(best_ckpt) else "final",
        "num_episodes": int(num_envs),
        "episode_steps": int(T),
        "total_reward_mean": rew_mean, "total_reward_ci95": rew_ci,
        "auc_mean": auc_mean, "auc_ci95": auc_ci,
    }
    if C_list:
        C_mean, C_ci = mean_confidence_interval(C_list)
        lbd_mean, lbd_ci = mean_confidence_interval(lbd_list)
        score_mean, score_ci = mean_confidence_interval(score_list)
        results.update({
            "overshoot_mean": C_mean, "overshoot_ci95": C_ci,
            "contraction_rate_mean": lbd_mean, "contraction_rate_ci95": lbd_ci,
            "contraction_score_mean": score_mean, "contraction_score_ci95": score_ci,
            "num_fit_groups": len(C_list),
        })

    print("[Eval] ── Best-model evaluation (non-terminating) ──")
    print(f"[Eval] total reward     : {rew_mean:.2f} ± {rew_ci:.2f} (95% CI, n={num_envs})")
    print(f"[Eval] error AUC        : {auc_mean:.4f} ± {auc_ci:.4f}")
    if C_list:
        print(f"[Eval] overshoot C      : {C_mean:.3f} ± {C_ci:.3f}")
        print(f"[Eval] contraction rate : {lbd_mean:.4f} ± {lbd_ci:.4f}  (C·e^(−λkΔt), min AUC)")

    out_json = os.path.join(agent.experiment_dir, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved → {out_json}")

    if not args_cli.no_wandb and "wandb" in sys.modules and sys.modules["wandb"].run is not None:
        wandb_logs = {}
        for k, v in results.items():
            if isinstance(v, (int, float)):
                if "reward" in k:
                    wandb_logs[f"Reward/{k}"] = v
                elif any(s in k for s in ["auc", "overshoot", "contraction_rate", "contraction_score"]):
                    wandb_logs[f"Stability/{k}"] = v
                else:
                    wandb_logs[f"final_eval/{k}"] = v
        sys.modules["wandb"].log(wandb_logs)


# ══════════════════════════════════════════════════════════════════════════════
# Classic route  (--classic flag)
# ══════════════════════════════════════════════════════════════════════════════
import gymnasium
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper


class BatchedGymnasiumWrapper(GymnasiumWrapper):
    """Overrides SKRL's default GymnasiumWrapper to prevent tensor-copy warnings.
    
    Our classic environments natively output PyTorch tensors for speed. SKRL's default
    wrapper forces torch.tensor() on the outputs, which throws a UserWarning in PyTorch
    when the output is already a tensor. This wrapper safely applies torch.as_tensor
    instead, fully implementing PyTorch's recommendation without modifying SKRL's library.
    """
    def step(self, actions: torch.Tensor):
        from skrl.utils.spaces.torch import (
            flatten_tensorized_space,
            tensorize_space,
            unflatten_tensorized_space,
            untensorize_space,
        )

        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()

        observation, reward, terminated, truncated, info = self._env.step(actions)

        # Convert to torch using .clone().detach() or as_tensor (implementing the PyTorch recommendation)
        observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation, device=self.device))

        # Here we fix the SKRL warning by checking if it's already a tensor!
        if torch.is_tensor(reward):
            reward = reward.clone().detach().to(self.device).view(self.num_envs, -1)
            terminated = terminated.clone().detach().to(self.device).view(self.num_envs, -1)
            truncated = truncated.clone().detach().to(self.device).view(self.num_envs, -1)
        else:
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
            terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
            truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)

        if self._vectorized:
            self._observation = observation
            self._info = info

        return observation, reward, terminated, truncated, info
