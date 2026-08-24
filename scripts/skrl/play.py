# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate trained checkpoints and report their Stability/* metrics.

Unlike a single-checkpoint viewer, this script is a multi-Model stability
evaluator: given one ``--task``, it discovers every trained model for that env
(one per algorithm) and rolls each out for a single episode across
``num_envs_for_eval`` parallel envs, reading the batched contraction metrics
(auc / contraction_rate / overshoot / contraction_score) from
``StatManagerEnvWrapper.stability_summary()`` — the same wrapper the training
loop logs from. Results are printed as a per-algorithm table and saved to
``logs/play/<env>_stability_<ts>.json``.

Model discovery (per algorithm), in order:
  1. ``<models_dir>/<env>/<alg>.pt``          (default ``models_dir="models"``)
  2. newest ``best_agent.pt`` under that algorithm's log tree
     (classic: ``logs/classic/<alg>/*/checkpoints/``;
      Isaac:   ``logs/skrl/*<alg>*/checkpoints/``)

Examples::

    # Classic — evaluate every algorithm trained on Car-v0
    python scripts/skrl/play.py --classic --task Car-v0

    # Classic — just one algorithm
    python scripts/skrl/play.py --classic --task Car-v0 --algorithm c3m

    # Isaac — evaluate every algorithm trained on the given task
    python scripts/skrl/play.py --task Quadruped-PathTracking-v0 --num_envs 64
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# ── Argument parsing ─────────────────────────────────────────────────────── #
# Parsed before any Isaac import so the classic route never touches Isaac Sim.

parser = argparse.ArgumentParser(description="Multi-model Stability evaluation of skrl checkpoints.")
parser.add_argument("--task", type=str, required=True, help="Name of the task/env.")
parser.add_argument(
    "--classic", action="store_true", default=False,
    help="Evaluate a classic env (no Isaac Sim launch). Mirrors train.py's --classic route.",
)
parser.add_argument(
    "--algorithm", type=str, default=None,
    help="Restrict to a single algorithm (e.g. c3m, ppo, c2rl-ppo). Default: every discovered model.",
)
parser.add_argument(
    "--models_dir", type=str, default="models",
    help="Preferred model root: <models_dir>/<env>/<alg>.pt. Falls back to logs/ if absent.",
)
parser.add_argument(
    "--num_envs", type=int, default=None,
    help="Parallel envs for the rollout (caps num_envs_for_eval). Default: algorithm-specific.",
)
parser.add_argument(
    "--num_envs_for_eval", type=int, default=64,
    help="Stability buffer size (envs whose trajectory is scored). Default 64.",
)
parser.add_argument("--seed", type=int, default=None, help="Environment/agent seed.")
parser.add_argument("--device", type=str, default=None, help="Compute device (default cuda:0 / cfg).")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (single algorithm).")
parser.add_argument(
    "--ref_length", "--ref-length", dest="ref_length", type=int, default=None,
    help="Reference-window length. Default AUTO from the cfg's discount_factor, matching "
         "train.py — override only if the run was trained with an explicit --ref_length.",
)
# Isaac-only extras (ignored on the classic route).
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Isaac: disable fabric.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])

_CONTRACTION_ALGOS = frozenset({"c3m", "lqr", "sdlqr", "cvstem-lqr", "c2rl-ppo", "c2rl-sac"})


def _env_name(task: str) -> str:
    """``Car-v0`` / ``Car-Play-v0`` -> ``Car`` (the <env> segment of models/<env>/)."""
    name = task.split(":")[-1]
    name = name.replace("-Play", "")
    # strip a trailing gymnasium version suffix (-v0, -v12, …)
    import re
    return re.sub(r"-v\d+$", "", name)


# ── Model discovery ──────────────────────────────────────────────────────── #

def _candidate_algorithms(classic: bool, algorithm_filter: str | None) -> list[str]:
    if algorithm_filter:
        return [algorithm_filter.lower()]
    if classic:
        root = os.path.join("logs", "classic")
        if os.path.isdir(root):
            return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        return []
    # Isaac: enumerate algorithms that ship a config entry point for this task
    # (best effort — the log-dir names aren't algorithm-clean like classic's).
    return ["ppo", "sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "c2rl-ppo", "c2rl-sac"]


def _newest_best_ckpt(patterns: list[str]) -> str | None:
    hits: list[str] = []
    for p in patterns:
        hits.extend(glob.glob(p, recursive=True))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _discover_models(task: str, classic: bool, models_dir: str,
                     algorithm_filter: str | None, checkpoint: str | None) -> list[tuple[str, str]]:
    """Return ``[(algorithm, checkpoint_path), …]`` for every model found for ``task``.

    Preference: ``<models_dir>/<env>/<alg>.pt``, else the newest ``best_agent.pt``
    under the algorithm's log tree. ``--checkpoint`` (with ``--algorithm``) forces
    a single explicit path.
    """
    env = _env_name(task)
    if checkpoint:
        alg = (algorithm_filter or "model").lower()
        return [(alg, os.path.abspath(checkpoint))]

    found: list[tuple[str, str]] = []
    for alg in _candidate_algorithms(classic, algorithm_filter):
        preferred = os.path.join(models_dir, env, f"{alg}.pt")
        if os.path.isfile(preferred):
            found.append((alg, os.path.abspath(preferred)))
            continue
        # skrl nests the checkpoints under an auto-named experiment subdir
        # (logs/classic/<alg>/<ts>/<..._AgentClass>/checkpoints/), so glob
        # recursively rather than assuming a fixed depth.
        if classic:
            patterns = [os.path.join("logs", "classic", alg, "**", "best_agent.pt")]
        else:
            patterns = [
                os.path.join("logs", "skrl", f"*{alg}*", "**", "best_agent.pt"),
                os.path.join("logs", "skrl", f"*{alg.replace('-', '_')}*", "**", "best_agent.pt"),
            ]
        ckpt = _newest_best_ckpt(patterns)
        if ckpt:
            found.append((alg, os.path.abspath(ckpt)))
    return found


# ── Rollout ──────────────────────────────────────────────────────────────── #

def _rollout_stability(env, agent, *, max_steps: int) -> dict:
    """Roll out one deterministic episode across all envs; return stability_summary().

    Mirrors C3MSkrlTrainer.eval(): reset (which invalidates any in-flight slots),
    step with the agent's deterministic action until every env has finished its
    episode (so the StatManagerEnvWrapper buffer completes and reduces to real
    metrics), then read the summary. Snapshots _compute_count to warn if the
    buffer never completed within the horizon.
    """
    import torch

    agent.enable_training_mode(False)
    for model in getattr(agent, "models", {}).values():
        if model is not None and hasattr(model, "eval"):
            model.eval()

    obs, _ = env.reset()
    states = env.state() if hasattr(env, "state") else None
    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", "cpu")
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    computes_before = getattr(env, "_compute_count", None)

    for _step in range(max_steps):
        with torch.no_grad():
            actions, outputs = agent.act(obs, states, timestep=0, timesteps=1)
            actions = outputs.get("mean_actions", actions) if isinstance(outputs, dict) else actions
        obs, _r, terminated, truncated, _ = env.step(actions)
        states = env.state() if hasattr(env, "state") else None
        finished |= (terminated | truncated).view(num_envs)
        if finished.all():
            break

    if computes_before is not None and getattr(env, "_compute_count", None) == computes_before:
        print("  [WARN] Stability buffer did not complete within the horizon — "
              "reported metrics may be stale/sentinel.")
    return env.stability_summary()


# ── Classic route ────────────────────────────────────────────────────────── #

def run_classic(args) -> list[dict]:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _src_dir = os.path.join(_root, "source", "c2rl")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    _classic_dir = os.path.join(_src_dir, "c2rl", "tasks", "direct")
    if _classic_dir not in sys.path:
        sys.path.insert(0, _classic_dir)
    sys.path.append(os.path.dirname(__file__))

    import c2rl.tasks.direct.classic  # noqa: F401 — registers classic envs
    import gymnasium as gym
    import yaml
    from c2rl.agents.skrl.contraction_metrics import StatManagerEnvWrapper
    from c2rl.agents.skrl.runner import CLActorRunner
    from c2rl.runners import ContractionRunner
    from train_utils import (
        BatchedGymnasiumWrapper,
        _default_num_envs_classic,
        _disarm_termination_for_eval,
        _inject_angle_idx,
        _resolve_symmetry_for_env,
    )

    def _load_agent_cfg(algorithm: str) -> dict:
        entry_key = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
        spec = gym.spec(args.task)
        entry = (spec.kwargs or {}).get(entry_key)
        if entry is None:
            raise ValueError(f"No '{entry_key}' registered for {args.task}.")
        pkg, fname = entry.split(":")
        import importlib
        pkg_obj = importlib.import_module(pkg)
        with open(os.path.join(os.path.dirname(pkg_obj.__file__), fname)) as f:
            return yaml.safe_load(f)

    models = _discover_models(args.task, True, args.models_dir, args.algorithm, args.checkpoint)
    if not models:
        print(f"[play] No models found for {args.task} (looked under {args.models_dir}/{_env_name(args.task)}/ "
              f"and logs/classic/*/).")
        return []

    device = args.device or "cuda:0"
    results: list[dict] = []
    for algorithm, ckpt in models:
        print(f"\n[play] === {algorithm} ===  ({ckpt})")
        agent_cfg = _load_agent_cfg(algorithm)
        agent_cfg["seed"] = args.seed if args.seed is not None else agent_cfg.get("seed", 42)
        # Eval-time config hygiene: no wandb, throwaway experiment dir, headless.
        exp = agent_cfg["agent"].setdefault("experiment", {})
        exp["wandb"] = False
        exp["directory"] = os.path.abspath(os.path.join("logs", "play", "_scratch"))
        exp["checkpoint_interval"] = 0
        exp["write_interval"] = 0
        agent_cfg.setdefault("trainer", {})
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["trainer"]["headless"] = True

        num_envs = args.num_envs if args.num_envs is not None else _default_num_envs_classic(algorithm)
        num_envs = max(num_envs, args.num_envs_for_eval)

        env = gym.make(args.task, num_envs=num_envs, device=device)
        # Playback is a measurement, same as train.py's evaluator — see
        # _disarm_termination_for_eval for why a truncated horizon inverts AUC.
        _disarm_termination_for_eval(env, "[play]")
        # Rebuild the same reference window the checkpoint was trained with.
        # train.py sizes it AUTO from gamma (RefWindow.length_for_horizon) unless
        # --ref_length was given, and its evaluator mirrors the training env's
        # window explicitly. play.py did neither, so it built a length-1 window
        # and every AUTO-sized checkpoint failed to load with a state_dict shape
        # mismatch (w2 18-wide in the checkpoint vs 9-wide here at gamma=0.01) —
        # i.e. this script could not evaluate any C2RL run at all.
        if hasattr(env.unwrapped, "configure_ref_window"):
            from c2rl.agents.skrl.ref_window import RefWindow as _RW
            _gamma = float(agent_cfg["agent"].get("discount_factor", 0.99))
            _offset = int(agent_cfg["agent"].get("ref_offset", 1) or 1)
            _len = args.ref_length if getattr(args, "ref_length", None) else _RW.length_for_horizon(
                _gamma, int(env.unwrapped.max_episode_len), _offset)
            env.unwrapped.configure_ref_window(length=_len, offset=_offset, gamma=_gamma)
        env = BatchedGymnasiumWrapper(env)
        env = StatManagerEnvWrapper(env, num_envs_for_eval=args.num_envs_for_eval)

        try:
            if algorithm in _CONTRACTION_ALGOS:
                if algorithm in ("c2rl-ppo", "c2rl-sac"):
                    agent_cfg["agent"]["use_empirical_dynamics"] = False
                runner = ContractionRunner(env, agent_cfg, task_id=args.task,
                                           num_envs=num_envs, is_classic=True)
            else:
                # Standalone PPO/SAC: strip the norm flags train.py handles and
                # inject angle_idx so the rebuilt models match the trained ones.
                _a = agent_cfg["agent"]
                _a.pop("use_state_norm", None)
                _a.pop("use_value_norm", None)
                for _k in ("state_preprocessor", "state_preprocessor_kwargs"):
                    _a.pop(_k, None)
                if algorithm == "ppo":
                    _a["value_preprocessor"] = "RunningStandardScaler"
                    _a["value_preprocessor_kwargs"] = None
                for _k in ("anneal_stddev", "anneal_log_std", "std_dev_annealing",
                           "std_dev_annealing_kwargs"):
                    _a.pop(_k, None)
                _angle_idx = list(getattr(env.unwrapped, "angle_idx", []) or [])
                _inject_angle_idx(agent_cfg, _angle_idx, _resolve_symmetry_for_env(env))
                runner = CLActorRunner(env, agent_cfg)

            print("  loading checkpoint …")
            runner.agent.load(ckpt)

            max_steps = int(getattr(env, "max_episode_length", getattr(env, "max_episode_len", 1000))) + 1
            summary = _rollout_stability(env, runner.agent, max_steps=max_steps)
        finally:
            env.close()

        _print_summary(algorithm, summary)
        results.append({"algorithm": algorithm, "checkpoint": ckpt, **summary})
    return results


# ── Isaac route ──────────────────────────────────────────────────────────── #

def run_isaac(args, simulation_app) -> list[dict]:
    import c2rl.tasks  # noqa: F401 — registers Isaac tasks
    import gymnasium as gym
    from c2rl.agents.skrl.contraction_metrics import StatManagerEnvWrapper
    from c2rl.agents.skrl.runner import CLActorRunner
    from c2rl.runners import ContractionRunner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry

    models = _discover_models(args.task, False, args.models_dir, args.algorithm, args.checkpoint)
    if not models:
        print(f"[play] No models found for {args.task}.")
        return []

    device = args.device
    results: list[dict] = []
    for algorithm, ckpt in models:
        print(f"\n[play] === {algorithm} ===  ({ckpt})")
        entry_key = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
        try:
            agent_cfg = load_cfg_from_registry(args.task, entry_key)
        except Exception as e:  # noqa: BLE001
            print(f"  [SKIP] no config entry '{entry_key}' for {args.task}: {e}")
            continue
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if device is not None:
            env_cfg.sim.device = device
        env_cfg.seed = args.seed if args.seed is not None else agent_cfg.get("seed", 42)

        agent_cfg.setdefault("agent", {}).setdefault("experiment", {})
        agent_cfg["agent"]["experiment"]["wandb"] = False
        agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(os.path.join("logs", "play", "_scratch"))
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["experiment"]["write_interval"] = 0

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = SkrlVecEnvWrapper(env, ml_framework=args.ml_framework)
        env = StatManagerEnvWrapper(env, num_envs_for_eval=args.num_envs_for_eval)

        try:
            _alg = algorithm.lower()
            if _alg in _CONTRACTION_ALGOS:
                agent_cfg["agent"]["use_empirical_dynamics"] = True
                runner = ContractionRunner(env, agent_cfg, is_classic=False)
            else:
                runner = CLActorRunner(env, agent_cfg)
            print("  loading checkpoint …")
            runner.agent.load(ckpt)

            max_steps = int(getattr(env, "max_episode_length", 1000)) + 1
            summary = _rollout_stability(env, runner.agent, max_steps=max_steps)
        finally:
            env.close()

        _print_summary(algorithm, summary)
        results.append({"algorithm": algorithm, "checkpoint": ckpt, **summary})
    return results


# ── Reporting ────────────────────────────────────────────────────────────── #

_METRIC_KEYS = ["auc_mean", "contraction_rate_mean", "running_lambda_mean",
                "overshoot_mean", "contraction_score_mean"]
# Short column labels so the table header lines up with the 16-wide value cells.
_METRIC_LABELS = {
    "auc_mean": "auc",
    "contraction_rate_mean": "lambda",
    "running_lambda_mean": "lambda_run",
    "overshoot_mean": "overshoot",
    "contraction_score_mean": "score",
}


def _print_summary(algorithm: str, summary: dict) -> None:
    if not summary:
        print(f"  {algorithm}: no stability metrics (buffer never completed).")
        return
    parts = [f"{_METRIC_LABELS[k]}={summary.get(k, float('nan')):.4g}" for k in _METRIC_KEYS]
    print(f"  {algorithm}: " + "  ".join(parts))


def _print_table(task: str, results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"Stability summary — {task}   (lower auc = better, higher score = better)")
    print("=" * 78)
    header = f"{'algorithm':<14}" + "".join(f"{_METRIC_LABELS[k]:>16}" for k in _METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['algorithm']:<14}"
        for k in _METRIC_KEYS:
            v = r.get(k)
            row += f"{v:>16.4g}" if isinstance(v, (int, float)) else f"{'—':>16}"
        print(row)
    print("=" * 78)


def _save_results(task: str, results: list[dict]) -> None:
    import json
    from datetime import datetime

    out_dir = os.path.join("logs", "play")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{_env_name(task)}_stability_{datetime.now():%Y-%m-%d_%H-%M-%S}.json")
    with open(out_path, "w") as f:
        json.dump({"task": task, "results": results}, f, indent=2)
    print(f"[play] Wrote {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────── #

def main() -> None:
    args_cli, _ = parser.parse_known_args()

    if args_cli.classic:
        results = run_classic(args_cli)
    else:
        # Launch Isaac Sim only for the Isaac route, and only after arg parsing.
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args_cli, _hydra = parser.parse_known_args()
        args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + " --/app/hangDetector/enabled=false"
        sys.argv = [sys.argv[0]] + _hydra
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        try:
            results = run_isaac(args_cli, simulation_app)
        finally:
            simulation_app.close()

    if results:
        _print_table(args_cli.task, results)
        _save_results(args_cli.task, results)


if __name__ == "__main__":
    main()
