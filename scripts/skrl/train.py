"""Train RL agents with skrl — Isaac Sim and classic gymnasium environments.

Isaac Sim (default):
    python scripts/skrl/train.py --task Quadruped-VelTracking-v0 --algorithm ppo
    python scripts/skrl/train.py --task Quadruped-PathTracking-v0 --algorithm c3m

Classic gymnasium (--classic flag, no Isaac Sim required):
    python scripts/skrl/train.py --classic --task Car-v0 --algorithm ppo
    python scripts/skrl/train.py --classic --task Car-v0 --algorithm c3m
"""

import argparse
import os
import sys

# Local wandb run files (config/history/media, not the same as tensorboard
# events) rack up one small file per run and are the #1 inode-quota killer on
# the cluster's home filesystem (~72k files observed, see search sweeps).
# ~/scratch has no file-count limit, so park them there instead — cloud
# syncing is unaffected. Must be set before wandb is ever imported.
os.environ.setdefault("WANDB_DIR", os.path.expanduser("~/scratch/wandb"))

# ─── Pre-parse: must know --classic before any Isaac Sim imports ──────────── #
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--classic", action="store_true", default=False)
_pre.add_argument("--task", type=str, default="")
_pre_args, _ = _pre.parse_known_args()
_is_classic = _pre_args.classic or _pre_args.task.startswith("classic")

if not _is_classic:
    from isaaclab.app import AppLauncher

# ─── Full argument parser ─────────────────────────────────────────────────── #
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--classic", action="store_true", default=False,
                    help="Use classic gymnasium environment (no Isaac Sim).")
parser.add_argument("--headon", action="store_true", default=False,
                    help="Show the Isaac Sim GUI. Isaac Sim runs headless by default; "
                         "pass this to disable headless mode and render a window.")
parser.add_argument("--task", type=str, default=None, help="Environment ID.")
parser.add_argument(
    "--algorithm", "--algo", type=str, default="PPO",
    help="Algorithm: ppo | sac | c3m | lqr | sdlqr | c2rl-ppo | c2rl-sac | AMP | DDPG | TD3 | …"
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--num_timesteps", "--num-timesteps", type=int, default=None,
                    help="Total training timesteps.")
parser.add_argument("--skip_final_eval", "--skip-final-eval", action="store_true", default=False,
                    help="Skip the post-training best-model evaluation rollout. That rollout is "
                         "sequential over a single env and does NOT feed the swept "
                         "Stability/* metrics (those come from the trainer loop), so sweeps "
                         "skip it — see search/build_sweep.py.")
parser.add_argument("--analytical", type=str, default="",
                    help="Pass 'dynamics' to use analytical dynamics (C3M/LQR).")
parser.add_argument("--use_empirical_dynamics", "--use-empirical-dynamics",
                    action="store_true", default=False,
                    help="Use a learned NeuralDynamics model instead of the env's exact analytical get_f_and_B "
                         "(classic envs only). When NOT passed, C3M/C2RL use analytical dynamics.")
parser.add_argument("--dynamics_checkpoint", "--dynamics-checkpoint", type=str, default=None,
                    help="Path to a dynamics.pt written by a previous C3M/C2RL run "
                         "(<log_dir>/checkpoints/dynamics.pt). REQUIRED for lqr/sdlqr/cvstem-lqr "
                         "on Isaac Sim envs: those algorithms train no dynamics model of their own "
                         "and Isaac envs expose no analytical get_f_and_B. Ignored by C3M/C2RL "
                         "(they own their dynamics) and unnecessary for classic envs.")
parser.add_argument("--caps_temporal_scale", "--caps-temporal-scale", type=float, default=None,
                    help="CAPS temporal action-smoothness weight on ||pi(s_t) - pi(s_t+1)||^2, added "
                         "to the POLICY LOSS (not the reward — the MDP/observation/dynamics are "
                         "untouched, so the CV-STEM certificate stays valid). Overrides the yaml "
                         "agent.caps_temporal_scale. 0 (default) disables it. "
                         "See agent_patches.patch_caps_regularizer.")
parser.add_argument("--caps_spatial_scale", "--caps-spatial-scale", type=float, default=None,
                    help="CAPS spatial action-smoothness weight on ||pi(s) - pi(s_bar)||^2 with "
                         "s_bar ~ N(s, caps_spatial_std^2) — penalizes policy state-gain rather "
                         "than chatter. Overrides the yaml agent.caps_spatial_scale.")
parser.add_argument("--caps_spatial_std", "--caps-spatial-std", type=float, default=None,
                    help="Sigma for the CAPS spatial perturbation, in the units the policy sees "
                         "in RAW observation units. Overrides the yaml agent.caps_spatial_std.")
parser.add_argument("--eig_reshape", "--eig-reshape", type=float, default=None,
                    help="ABLATION (c2rl_ppo classic only): reshape the Mahalanobis reward's M "
                         "eigenvalue SPREAD to this target cond(M), keeping eigenvectors and "
                         "geometric-mean scale fixed — isolates conditioning from what the C1/C2 "
                         "fit converged to. See env_base.py's set_eig_reshape.")

# W&B
parser.add_argument("--no_wandb", "--no-wandb", action="store_true", default=False,
                    help="Disable Weights & Biases logging.")
parser.add_argument("--wandb_project", "--wandb-project", type=str, default="c2rl",
                    help="W&B project name.")
parser.add_argument("--wandb_run_name", "--wandb-run-name", type=str, default=None,
                    help="W&B run name.")
parser.add_argument("--wandb_batch", "--wandb-batch", type=str, default=None,
                    help="Free-text label grouping this launch, recorded to wandb.config as "
                         "`run_batch` so one relaunch is filterable/groupable apart from the "
                         "previous one. Defaults to the run date (YYYY-MM-DD), which already "
                         "separates today's runs from yesterday's with no flag.")

# Isaac Sim-specific
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training (Isaac only).")
parser.add_argument("--video_length", type=int, default=0,
                    help="Length of video in steps (0 = auto-calculate to 1 episode length)")
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--agent", type=str, default=None,
                    help="Explicit skrl cfg entry-point key (Isaac only).")
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
parser.add_argument("--debug_vis", action="store_true", default=False)
# ── Agent-config overrides — flag name == YAML key under `agent:` ─────────── #
# Each dest is spelled exactly like the key it writes in agents/skrl_*_cfg.yaml,
# so `--discount_factor 0.1` does what editing `discount_factor: 0.1` would.
# Applied to agent_cfg["agent"] in both the classic and Isaac branches via
# _apply_agent_overrides (previously only the Isaac branch honored them, so a
# classic run silently ignored everything but --learning_rate/--discount_factor).
# The older --ppo_*/--sac_*/--lr/--epochs spellings are kept as aliases on the
# same arguments, so existing commands and --extra strings keep working.
_ov = parser.add_argument_group("agent config overrides (flag == YAML key)")
# Shared by PPO and SAC.
_ov.add_argument("--learning_rate", "--learning-rate", "--lr", "--ppo_lr", "--ppo-lr",
                 "--sac_lr", "--sac-lr", type=float, default=None)
_ov.add_argument("--discount_factor", "--discount-factor", "--ppo_discount", "--ppo-discount",
                 "--sac_discount", "--sac-discount", type=float, default=None)
# PPO.
_ov.add_argument("--rollouts", "--ppo_rollouts", "--ppo-rollouts", type=int, default=None)
_ov.add_argument("--learning_epochs", "--learning-epochs", "--epochs",
                 "--ppo_learning_epochs", "--ppo-learning-epochs", type=int, default=None)
_ov.add_argument("--mini_batches", "--mini-batches", "--ppo_mini_batches", "--ppo-mini-batches",
                 type=int, default=None)
# skrl PPO's GAE key is `lambda`; the yaml also carries `gae_lambda` — set both.
_ov.add_argument("--gae_lambda", "--gae-lambda", "--lambda", "--ppo_lambda", "--ppo-lambda",
                 type=float, default=None)
_ov.add_argument("--ratio_clip", "--ratio-clip", "--ppo_ratio_clip", "--ppo-ratio-clip",
                 type=float, default=None)
_ov.add_argument("--entropy_loss_scale", "--entropy-loss-scale",
                 "--ppo_entropy_scale", "--ppo-entropy-scale", type=float, default=None)
_ov.add_argument("--kl_threshold", "--kl-threshold", "--ppo_kl_threshold", "--ppo-kl-threshold",
                 type=float, default=None)
# AUC-aligned Euclidean-decrement reward (see c2rl.C2RLPPOCfg.reward_euclidean).
# Works for standalone PPO/SAC too (applied straight to the env — see the
# `not _is_contraction` block right after raw_env is built) as well as C2RL.
_ov.add_argument("--reward_euclidean", "--reward-euclidean",
                 dest="reward_euclidean", action="store_true", default=None)
_ov.add_argument("--reward_level", "--reward-level",
                 dest="reward_level", action="store_true", default=None)
# End the episode when the state leaves the env's X_TERMINATION_* box, instead of
# silently pinning it there for the rest of the horizon (env_base.step's clamp).
# On by default (env_base) — `--no_terminate_out_of_box` restores the old
# never-terminating behaviour, which is what any pre-flip number was measured
# under. None means "not specified", so the env keeps its own default.
parser.add_argument("--terminate_out_of_box", "--terminate-out-of-box",
                    dest="terminate_out_of_box", action="store_true", default=None)
parser.add_argument("--no_terminate_out_of_box", "--no-terminate-out-of-box",
                    dest="terminate_out_of_box", action="store_false")
# Report that excursion on `terminated` rather than `truncated`. Zeroes the GAE
# bootstrap, which on a cost reward is a suicide bonus — see
# BaseEnv.set_terminate_out_of_box.__doc__ before using it.
parser.add_argument("--terminate_as_terminal", "--terminate-as-terminal",
                    dest="terminate_as_terminal", action="store_true", default=False)
# C2RL only: penalise ||u - uref||^2 in the reward (residual trust anchor).
_ov.add_argument("--residual_anchor_scale", "--residual-anchor-scale",
                 dest="residual_anchor_scale", type=float, default=None)
_ov.add_argument("--pretrain_critic_steps", "--pretrain-critic-steps",
                 dest="pretrain_critic_steps", type=int, default=None)
_ov.add_argument("--pretrain_critic_epochs", "--pretrain-critic-epochs",
                 dest="pretrain_critic_epochs", type=int, default=None)
_ov.add_argument("--pretrain_critic_lr", "--pretrain-critic-lr",
                 dest="pretrain_critic_lr", type=float, default=None)
_ov.add_argument("--disable_advantage_norm", "--disable-advantage-norm",
                 dest="disable_advantage_norm", action="store_true", default=None)
_ov.add_argument("--value_loss_scale", "--value-loss-scale", type=float, default=None)
_ov.add_argument("--grad_norm_clip", "--grad-norm-clip", type=float, default=None)
_ov.add_argument("--use_state_norm", "--use-state-norm", "--ppo_use_state_norm",
                 "--ppo-use-state-norm", type=str, default=None)
_ov.add_argument("--use_value_norm", "--use-value-norm", "--ppo_use_value_norm",
                 "--ppo-use-value-norm", type=str, default=None)
# Its partner knob (rl_glue installs them through different hooks:
# value_preprocessor vs rewards_shaper), so an A/B on the reward normalizer
# needs its own flag — without one the only way to toggle it is editing the
# yaml, which is shared by every concurrently-running sweep worker.
_ov.add_argument("--use_reward_norm", "--use-reward-norm", "--ppo_use_reward_norm",
                 "--ppo-use-reward-norm", type=str, default=None)
# "false" makes PPO learn log_std through the policy loss instead of following
# the fixed std_dev_annealing_kwargs schedule (C2RL-PPO only; see C2RLPPOCfg).
_ov.add_argument("--std_dev_annealing", "--std-dev-annealing", type=str, default=None)
# Train on a fixed set of num_envs reference trajectories (one minted per env
# slot on its first reset) instead of drawing a fresh one every episode. The
# initial error stays random — this fixes the task, not the initial condition.
_ov.add_argument("--fix_ref_trajectories", "--fix-ref-trajectories",
                 action="store_true", default=False)
# SAC.
_ov.add_argument("--batch_size", "--batch-size", "--sac_batch_size", "--sac-batch-size",
                 type=int, default=None)
_ov.add_argument("--polyak", "--sac_polyak", "--sac-polyak", type=float, default=None)
_ov.add_argument("--gradient_steps", "--gradient-steps", "--sac_gradient_steps",
                 "--sac-gradient-steps", type=int, default=None)
_ov.add_argument("--initial_entropy_value", "--initial-entropy-value", "--sac_entropy",
                 "--sac-entropy", type=float, default=None)
# Under the `memory:` block, not `agent:`.
_ov.add_argument("--memory_size", "--memory-size", "--sac_memory_size", "--sac-memory-size",
                 type=int, default=None)
# Model-structure overrides (manipulate the `models:` block, not `agent:`).
parser.add_argument("--ppo_activations", "--ppo-activations", type=str, default=None)
parser.add_argument("--ppo_network_arch", "--ppo-network-arch", type=str, default=None)

# ── Reference window: the observation s = {x, xrefs, urefs} ──────────────── #
# xrefs[k] = xref[t + k*ref_offset], k = 0..ref_length-1 (k=0 is the current
# reference). This is the POMDP fix that makes a high discount_factor valid:
# V(s) integrates the reward over ~1/(1-gamma) steps, so every reference point
# inside that horizon must be in the observation. RefWindow.check_markov warns
# at env construction when the window is too short for the discount.
# Default AUTO: the length is derived from this run's discount_factor so the
# window always spans the effective horizon 1/(1-gamma) — see
# RefWindow.length_for_horizon. Sizing it by hand is what let a window sized for
# one gamma be used at another, silently making V non-Markov (or 50x oversized)
# with nothing in the logs saying so. Pass an explicit integer to override.
parser.add_argument("--ref_length", "--ref-length", type=int, default=None,
                    help="Reference points in the observation window. Default: AUTO — "
                         "sized so the window spans the discount's effective horizon "
                         "1/(1-discount_factor). Pass an int to pin it.")
parser.add_argument("--ref_offset", "--ref-offset", type=int, default=1,
                    help="Step stride between window points: xrefs[k] = xref[t + k*OFFSET]. "
                         "Widens the span per point, at the cost of subsampling the "
                         "horizon (>1 is reported as non-Markov — see RefWindow.check_markov).")

# How both the actor's W2(xrefs) and the critic's psi(xrefs) turn the reference
# window into a fixed vector. Shared by design — the critic's independence comes
# from its own architecture (phi/psi/combine), not a second encoder choice.
# default=None, not "mlp"/1, so "explicitly passed" is distinguishable from "not
# passed". With a non-None default the apply site could only use setdefault (a
# plain assignment would clobber a sweep-sampled value back to the CLI default
# every trial) -- and setdefault means the YAML silently wins over an explicit
# flag. Measured: `--encoder attn --encoder_stride 1` on classic-car-v0, whose
# yaml pins encoder: gru / stride: 3, ran as gru/3 and said nothing, so a VRAM
# measurement taken that way was of the wrong configuration entirely.
parser.add_argument("--encoder", "--enc", type=str, default=None,
                    choices=["mlp", "gru", "attn"],
                    help="Reference-window encoder for the actor's W2 and the critic's "
                         "psi: mlp (flatten), gru (recency-weighted), attn (learned "
                         "attention over points). Overrides the yaml. Default when "
                         "absent: the yaml's value, else mlp.")
parser.add_argument("--encoder_stride", "--encoder-stride", type=int, default=None,
                    help="Keep every Nth window point (nearest-first) before encoding; "
                         "1 = dense. Bounds 'mlp' input width on a long window. "
                         "Overrides the yaml; default when absent: the yaml's value, "
                         "else 1.")
parser.add_argument("--critic_encoder", "--critic-encoder", type=str, default=None,
                    choices=["mlp", "gru", "attn"],
                    help="Override the critic's psi encoder (defaults to --encoder).")
parser.add_argument("--critic_encoder_stride", "--critic-encoder-stride", type=int, default=None,
                    help="Stride for the critic's psi encoder (see --encoder_stride).")
parser.add_argument("--critic_combine", "--critic-combine", type=str, default="concat",
                    choices=["concat"],
                    help="How the critic combines phi(x, e) and psi(xrefs). Only "
                         "'concat' is active; 'bilinear'/'film' are commented out "
                         "in models.py.")
# O6: parameterize the critic as V(s) = f_theta(s) + ||e||^2_M, with the second
# term computed analytically from the frozen CMG instead of learned. The
# decrement reward's telescoping identity is V_shaped = (1-gamma)V_orig - Phi(s),
# so the O(1) potential the shaping removes from the reward reappears in the
# critic's target; adding it back in closed form leaves f_theta only the O(dt)
# part. Pair with --use_value_norm false (the term is in real value units).
parser.add_argument("--critic_analytic_potential", "--critic-analytic-potential",
                    dest="critic_analytic_potential", action="store_true", default=False,
                    help="Critic V(s) = f_theta(s) + ||e||^2_M using the frozen CMG "
                         "(see models._AnalyticPotentialMixin). Use with use_value_norm=false.")
parser.add_argument("--critic_embed_dim", "--critic-embed-dim", type=int, default=64,
                    help="Embedding width for the privileged critic's phi/psi "
                         "(default 64). Bounds the bilinear form's rank.")

# UVFA-style generalization test: Also evaluate the trained policy on a fixed
# bank of reference-trajectory shapes drawn from a generator seeded
# independently of training (see env.CarEnv.set_held_out_mode) — guaranteed
# never encountered during training, unlike the normal post-training eval's
# i.i.d.-random trajectories (a different draw from the same distribution).
# None (default) = skip this second eval pass entirely.
parser.add_argument("--eval_held_out_seed", "--eval-held-out-seed", type=int, default=None,
                    help="Generator seed for a FIXED, held-out bank of reference-"
                         "trajectory shapes (must differ from --seed). None = skip "
                         "this extra generalization eval.")
parser.add_argument("--eval_held_out_trajectories", "--eval-held-out-trajectories",
                    type=int, default=64,
                    help="Size of the fixed held-out reference-trajectory bank.")

# Classic-specific
parser.add_argument("--cfg", type=str, default=None,
                    help="Path to a custom YAML config (classic only).")
# The Isaac branch gets --device from AppLauncher; the classic branch has no
# AppLauncher, so register it here (gated to avoid a conflicting option string
# with AppLauncher's --device). Default None -> cuda:0 if available else cpu.
if _is_classic:
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device for classic runs (default: cuda:0 if "
                             "available, else cpu).")

# Reference trajectory generation (auto-triggered after vel-tracking training)
parser.add_argument("--ref_num_trajs", type=int, default=1000,
                    help="Number of reference trajectories to collect after vel-tracking training.")
parser.add_argument("--min_ref_quality", type=float, default=None,
                    help="Minimum mean episode reward before generating ref trajs. 0 to skip check.")
parser.add_argument("--min_ref_traj_length_frac", type=float, default=0.5,
                    help="Minimum fraction of the max episode length (T) a trajectory must survive "
                         "to be accepted into dynamics_data.npz. Trajectories shorter than "
                         "min_ref_traj_length_frac * T are discarded (default 0.5, i.e. half of T).")
parser.add_argument("--ref_oversample_factor", type=float, default=2.0,
                    help="Collect this many times ref_num_trajs candidate trajectories (that clear "
                         "min_ref_traj_length_frac) before keeping only the longest ref_num_trajs of "
                         "them — gives the selection room to prefer complete rollouts over ones that "
                         "survived just past the minimum length. 1.0 disables oversampling.")

if not _is_classic:
    AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if not _is_classic:
    args_cli.headless = not args_cli.headon
if not _is_classic and args_cli.video:
    args_cli.enable_cameras = True
    if "--enable_cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")

if not _is_classic:
    args_cli.kit_args = (args_cli.kit_args or "") + " --/app/hangDetector/enabled=false"
    hydra_args = [arg for arg in hydra_args if not (arg.startswith("--") and ("=" in arg or "." in arg))]
    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Suppress noisy mesh/hydra warnings from Isaac Sim assets
    import carb
    carb.settings.get_settings().set("/log/logger/channelFilter", "-omni.hydra")

# ─── Shared imports ───────────────────────────────────────────────────────── #
import logging
import os
import random
from datetime import datetime

import gymnasium as gym
import yaml
from train_utils import (
    _default_num_envs_classic,
    _evaluate_best_model,
    _evaluate_classic_path_tracking,
    _generate_ref_trajs,
    _inject_angle_idx,
    _resolve_caps_kwargs,
    _resolve_symmetry_for_env,
    apply_agent_patches,
    apply_cli_dotted_overrides,
    apply_wandb_sweep_overrides,
    disable_tensorboard_files,
    finish_wandb,
    install_wandb_scalar_hook,
    normalize_agent_cfg,
)

# No env's tensorboard event files are ever read back by anything in this repo
# (all dashboards are wandb) — disable disk writes unconditionally, for both
# --classic and Isaac Sim routes, regardless of whether wandb is enabled.
disable_tensorboard_files()

algorithm = args_cli.algorithm.lower()
# Bare "c2rl" (no -ppo/-sac suffix) defaults to the PPO variant, since it has
# no standalone (non-suffixed) config entry point registered.
if algorithm in ("c2rl",):
    algorithm = f"{algorithm}_ppo"
_CONTRACTION_ALGOS = {
    "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr",
    "c2rl-ppo", "c2rl-sac", "c2rl_ppo", "c2rl_sac",
}

# Algorithm-aware num_envs defaults (used when --num_envs is not given).
# SAC-based algorithms (and c3m/lqr/sdlqr, which sample from a large buffer the
# same way SAC does) need far fewer parallel envs; PPO-based algorithms are
# on-policy and benefit from massively parallel envs. Applies to both the
# classic gymnasium route and the Isaac Sim route.
_SAC_LIKE_ALGOS = {"sac", "c2rl-sac", "c2rl_sac", "c3m", "lqr", "sdlqr", "cvstem-lqr", "cvstem_lqr"}
_DEFAULT_NUM_ENVS_SAC = 64
_DEFAULT_NUM_ENVS_PPO_CLASSIC = 1024


def _apply_agent_overrides(agent_cfg, args):
    """Write every set agent-config override into agent_cfg, by YAML key name.

    Shared by the classic and Isaac branches so a flag behaves identically on
    both (before, only the Isaac branch applied most of them). Each flag's dest
    is the YAML key it targets; ``None`` means "not passed, leave the config's
    value". ``lambda``/``gae_lambda`` and ``memory_size`` are the only keys whose
    destination differs from a plain ``agent[<dest>]`` write.
    """
    a = agent_cfg["agent"]
    simple = (
        "learning_rate", "discount_factor", "rollouts", "learning_epochs",
        "mini_batches", "ratio_clip", "entropy_loss_scale", "kl_threshold",
        "value_loss_scale", "grad_norm_clip", "batch_size", "polyak",
        "gradient_steps", "initial_entropy_value",
    )
    for key in simple:
        val = getattr(args, key, None)
        if val is not None:
            a[key] = val
    if args.gae_lambda is not None:
        a["lambda"] = args.gae_lambda      # skrl PPO's GAE key
        a["gae_lambda"] = args.gae_lambda  # yaml's own spelling
    for key in ("use_state_norm", "use_value_norm", "use_reward_norm", "std_dev_annealing"):
        val = getattr(args, key, None)
        if val is not None:
            a[key] = (str(val).lower() == "true")
    # One switch, two places: the agent flag controls the annealing patch, while
    # the CLActor must also be built with a learnable logstd. Setting only the
    # former leaves self.anneal=True on the model, so CLActor.anneal_stddev()
    # would still overwrite whatever the policy gradient learned.
    if getattr(args, "std_dev_annealing", None) is not None:
        _pol = agent_cfg.get("models", {}).get("policy")
        if isinstance(_pol, dict):
            _pol["anneal_stddev"] = a["std_dev_annealing"]
    if args.memory_size is not None:
        agent_cfg["memory"]["memory_size"] = args.memory_size
    # Encoder settings are model kwargs (models.policy.* / models.critic.*), not
    # agent-config keys — contraction_runner passes those blocks through verbatim
    # as model kwargs. setdefault, never assignment: a wandb sweep parameter under
    # models.policy.* already landed in agent_cfg via apply_wandb_sweep_overrides,
    # and a plain assignment would silently clobber it back to the CLI default on
    # every trial.
    _policy_block = agent_cfg.get("models", {}).get("policy")
    if isinstance(_policy_block, dict):
        # Explicit flag wins over the yaml; absent flag leaves the yaml (or a
        # sweep-sampled value) alone and falls back to mlp/1 only if neither set
        # it. Both halves matter: assignment-always clobbers sweeps, and
        # setdefault-always makes the flag a no-op on any env that pins these.
        if args.encoder is not None:
            _policy_block["encoder"] = args.encoder
        else:
            _policy_block.setdefault("encoder", "mlp")
        if args.encoder_stride is not None:
            _policy_block["encoder_stride"] = args.encoder_stride
        else:
            _policy_block.setdefault("encoder_stride", 1)
    if getattr(args, "reward_euclidean", None):
        a["reward_euclidean"] = True
    if getattr(args, "reward_level", None):
        a["reward_level"] = True
    if getattr(args, "residual_anchor_scale", None) is not None:
        a["residual_anchor_scale"] = args.residual_anchor_scale
    if getattr(args, "pretrain_critic_steps", None) is not None:
        a["pretrain_critic_steps"] = args.pretrain_critic_steps
    if getattr(args, "pretrain_critic_epochs", None) is not None:
        a["pretrain_critic_epochs"] = args.pretrain_critic_epochs
    if getattr(args, "pretrain_critic_lr", None) is not None:
        a["pretrain_critic_lr"] = args.pretrain_critic_lr
    if getattr(args, "disable_advantage_norm", None):
        a["disable_advantage_norm"] = True


seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 10000)

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_VEL_TASK_TO_ROBOT = {"Quadruped": "quadruped", "Humanoid": "humanoid", "Manipulator": "manipulator"}


if _is_classic:
    import os as _os
    import sys as _sys

    import numpy as np
    import torch

    # Register classic envs by importing the classic package
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _classic_dir = os.path.join(
        _root, "source", "c2rl", "c2rl", "tasks", "direct",
    )
    if _classic_dir not in sys.path:
        sys.path.insert(0, _classic_dir)
    import c2rl.tasks.direct.classic  # noqa: F401 — registers gymnasium envs (e.g. Car-v0)

    # ── Config loading ────────────────────────────────────────────────────── #
    def _load_cfg(entry_point_key: str, custom_path: str | None = None) -> dict:
        if custom_path:
            with open(custom_path) as f:
                return yaml.safe_load(f)
        spec = gym.spec(args_cli.task)
        kwargs = spec.kwargs or {}
        entry = kwargs.get(entry_point_key)
        if entry is None:
            raise ValueError(
                f"No '{entry_point_key}' registered for {args_cli.task}. "
                f"Available: {list(kwargs.keys())}"
            )
        pkg, fname = entry.split(":")
        import importlib
        pkg_obj = importlib.import_module(pkg)
        cfg_path = os.path.join(os.path.dirname(pkg_obj.__file__), fname)
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    entry_key = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
    agent_cfg = _load_cfg(entry_key, args_cli.cfg)
    # Hand-pinned `--agent.mini_batches=16`-style knobs, applied the same way a
    # sweep would. Before the seed read and every model build below, so an
    # overridden value is what actually gets used. A wandb sweep still wins: it
    # applies later (search "apply_wandb_sweep_overrides"), which is correct —
    # the sweep is the one sampling that axis.
    apply_cli_dotted_overrides(agent_cfg, hydra_args)
    # --seed CLI arg wins; otherwise fall back to the yaml's own seed (not the
    # random.randint(...) module-level `seed` computed at line 164 before the
    # yaml was even loaded — using that unconditionally silently discarded
    # every config's `seed:` field and made "the same command" produce a
    # different random init/data-sampling trajectory on every invocation).
    # Mirrors the Isaac-env branch below (search "env_cfg.seed"), which
    # already got this right.
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent_cfg["seed"] = seed

    if args_cli.num_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
    # All agent-config overrides (--learning_rate, --discount_factor, --rollouts,
    # …) — the classic branch used to apply only a couple of these, silently
    # dropping the rest, so a classic gamma sweep trained at the yaml's own value.
    _apply_agent_overrides(agent_cfg, args_cli)
    # Classic contraction envs use the env's exact analytical get_f_and_B by
    # default (use_empirical_dynamics=False); pass --use_empirical_dynamics to
    # learn a NeuralDynamics instead. Classic envs only (Isaac forces empirical).
    # _CONTRACTION_ALGOS (not a hand-written list): it carries both the hyphen
    # and underscore spellings of the c2rl variants. The old inline list had
    # only "c2rl_ppo"/"c2rl_sac", so the README-documented "--algorithm c2rl-ppo"
    # silently dropped --use_empirical_dynamics and trained on analytical
    # dynamics instead.
    if algorithm in _CONTRACTION_ALGOS:
        agent_cfg["agent"]["use_empirical_dynamics"] = args_cli.use_empirical_dynamics
    if args_cli.dynamics_checkpoint:
        agent_cfg["agent"]["dynamics_checkpoint"] = args_cli.dynamics_checkpoint

    _run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "classic", algorithm, _run_ts)
    os.makedirs(log_dir, exist_ok=True)
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(log_dir)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["trainer"]["headless"] = True

    # Sweep trials are throwaway: don't let hundreds of trials each write
    # ~10 full checkpoints to logs/ (see play.py:239,327 for the same idiom).
    if "WANDB_SWEEP_ID" in os.environ:
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # W&B
    if not args_cli.no_wandb:
        import wandb as _wandb
        agent_cfg["agent"]["experiment"]["wandb"] = ("WANDB_SWEEP_ID" not in os.environ)
        _wkw = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
        _wkw["project"] = args_cli.wandb_project
        _wkw["sync_tensorboard"] = False
        # Force console capture even when stdout isn't a tty — sweep agents are
        # launched backgrounded with stdout/stderr redirected to a logfile
        # (commands/search.sh: `wandb agent ... > logfile 2>&1 &`), which
        # is exactly the case where wandb's tty auto-detection for the Logs tab
        # can silently fail to capture anything. "wrap" forces it regardless.
        try:
            _wkw["settings"] = _wandb.Settings(console="wrap")
        except Exception:
            pass
        # Consistent run name: CLI override > YAML-provided name > deterministic
        # default that matches the local log directory (logs/classic/<algo>/<ts>).
        # env + algorithm + seed + date. The old default was
        # f"classic_{algorithm}_{_run_ts}", which named every env identically and
        # omitted the seed, so a 10-seed x 3-gamma x 4-env batch arrived in W&B as
        # 120 same-titled runs distinguishable only by opening each one's config.
        # args_cli.task keeps its version suffix ("classic-car-v0" -> "car-v0",
        # "classic-car-v1" -> "car-v1"): stripping "-v0" would collapse the two car
        # variants this batch exists to contrast.
        _env_tag = (args_cli.task or "env").removeprefix("classic-")
        _wkw["name"] = (args_cli.wandb_run_name or _wkw.get("name")
                        or f"{_env_tag}_{algorithm}_seed{seed}_{_run_ts}")
        # wandb.run.name is not a config key, so it can't be filtered/grouped on
        # in the runs table — mirror it into the config, alongside a batch label
        # that defaults to the run date so a relaunch is separable from the
        # previous day's runs without passing anything.
        _wkw.setdefault("config", {}).update({
            "run_name": _wkw["name"],
            "run_batch": args_cli.wandb_batch or _run_ts.split("_")[0],
        })

        # A sweep must init early: its sampled hyperparameters only exist on
        # wandb.config once the run is live, and they have to reach agent_cfg
        # before any model is built from it.
        if "WANDB_SWEEP_ID" in os.environ:
            if _wandb.run is None:
                _wandb.init(project=_wkw["project"], name=_wkw.get("name"),
                            sync_tensorboard=False, settings=_wkw.get("settings"))
            apply_wandb_sweep_overrides(agent_cfg)

        install_wandb_scalar_hook()
    else:
        # --no_wandb: override the YAML default (wandb: true) so the skrl agent
        # does not call wandb.init() during agent.init().
        agent_cfg["agent"].setdefault("experiment", {})["wandb"] = False

    _src_dir = os.path.join(_root, "source", "c2rl")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    _sys.path.append(_os.path.dirname(__file__))

    from c2rl.agents.skrl.contraction_metrics import StatManagerEnvWrapper
    from train_utils import BatchedGymnasiumWrapper
    from wandb_plot_wrapper import WandbPlotWrapper

    _is_contraction = algorithm in _CONTRACTION_ALGOS
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else _default_num_envs_classic(algorithm)
    device = getattr(args_cli, "device", None)
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    raw_env = gym.make(args_cli.task, num_envs=num_envs, device=device)
    # Standalone PPO/SAC path: apply the euclidean/level reward switch straight
    # to the env — C2RL applies the equivalent through its own set_ccm() call
    # inside ContractionRunner (see c2rl.py), and normalize_agent_cfg pops these
    # two keys out of agent_cfg["agent"] for ppo/sac so they never reach the
    # real skrl PPO_CFG/SAC_CFG (which would reject them as unknown fields).
    if not _is_contraction and (args_cli.reward_euclidean or args_cli.reward_level):
        raw_env.unwrapped.reward_euclidean = bool(args_cli.reward_euclidean)
        raw_env.unwrapped.reward_level = bool(args_cli.reward_level)
    # None = not specified, so the env keeps its own default (on). Only an
    # explicit flag reaches through here.
    if args_cli.terminate_out_of_box is not None or args_cli.terminate_as_terminal:
        if not hasattr(raw_env.unwrapped, "set_terminate_out_of_box"):
            raise SystemExit("--terminate_out_of_box requires an env_base/PathTrackingBase env (got "
                             f"{type(raw_env.unwrapped).__name__})")
        raw_env.unwrapped.set_terminate_out_of_box(
            True if args_cli.terminate_out_of_box is None else args_cli.terminate_out_of_box,
            terminal=bool(args_cli.terminate_as_terminal))
    if args_cli.eig_reshape is not None:
        if not hasattr(raw_env.unwrapped, "set_eig_reshape"):
            raise SystemExit("--eig_reshape requires a classic env_base env (got "
                             f"{type(raw_env.unwrapped).__name__})")
        raw_env.unwrapped.set_eig_reshape(args_cli.eig_reshape)
        print(f"[train] eig_reshape ACTIVE: Mahalanobis reward's M reshaped to "
              f"cond(M) = {args_cli.eig_reshape:g} every step")

    # Reference window: size it for this run and rebuild observation_space.
    # Must run before the wrappers/runner so the new space propagates to the
    # models and the memory. discount_factor is already finalized here (CLI
    # overrides at _apply_agent_overrides + any wandb-sweep overrides above), so
    # the Markov check below reports against the gamma actually used.
    _gamma = float(agent_cfg["agent"].get("discount_factor", 0.99))
    if hasattr(raw_env.unwrapped, "configure_ref_window"):
        # Imported unconditionally: the window CONFIG block below calls
        # _RW.effective_horizon() on both paths, and it used to be imported only
        # inside the AUTO branch -- so every pinned run died with
        # "NameError: name '_RW' is not defined" before reaching training.
        from c2rl.agents.skrl.ref_window import RefWindow as _RW
        _len = args_cli.ref_length
        if _len is None:
            # Default is the FULL episode, the same length at every gamma, with
            # the tail repeating the last reference point (RefWindow clamps
            # indices past the episode end).
            #
            # Sizing it from gamma instead -- which is what this did -- makes the
            # observation a function of gamma: 2 points at gamma=0.01, 101 at
            # 0.99, 500 at 0.999. A gamma sweep run that way varies gamma AND the
            # observation width together, so any effect it measures is
            # gamma-and-architecture with no way to separate them afterwards. It
            # also made the cells wildly unequal in cost (6.87 it/s at gamma=0.01
            # against 0.41 it/s at 0.999, a 16.5x spread) so the high-gamma cells
            # could not finish inside a wall the low-gamma ones cleared easily.
            #
            # A full window is never too short for the discount, so check_markov
            # cannot fire; at low gamma it is larger than the horizon needs,
            # which costs compute but confounds nothing.
            _len = int(raw_env.unwrapped.max_episode_len) // max(args_cli.ref_offset, 1)
            print(f"[train] ref_length FULL -> {_len} "
                  f"(episode {int(raw_env.unwrapped.max_episode_len)} steps, "
                  f"offset={args_cli.ref_offset}; gamma={_gamma} does NOT change it. "
                  f"Effective horizon at this gamma is "
                  f"{_RW.effective_horizon(_gamma, int(raw_env.unwrapped.max_episode_len))} steps)")
        raw_env.unwrapped.configure_ref_window(
            length=_len, offset=args_cli.ref_offset, gamma=_gamma)
        # The window reaches wandb as CONFIG. Without it the gamma/window pairing
        # is invisible in the logged data: ref_length AUTO makes the observation
        # width a function of gamma (19-wide at 0.5, 2504-wide at 0.999), so any
        # gamma effect read off runs that do not record their window is really a
        # gamma-and-architecture effect with no way to tell after the fact.
        # ref_length_mode says whether this run pinned it or inherited AUTO.
        _mel = int(raw_env.unwrapped.max_episode_len)
        _win_cfg = {
            "ref_length": int(_len),
            "ref_offset": int(args_cli.ref_offset),
            "ref_length_mode": "pinned" if args_cli.ref_length is not None else "full",
            "ref_window_span": int(_len) * int(args_cli.ref_offset),
            "effective_horizon": int(_RW.effective_horizon(_gamma, _mel)),
            "max_episode_len": _mel,
            # <1 means the window is shorter than the discount's horizon, i.e. V
            # is non-Markov in the reference -- the POMDP the window exists to
            # prevent.
            "window_over_horizon": (int(_len) * int(args_cli.ref_offset))
            / max(int(_RW.effective_horizon(_gamma, _mel)), 1),
        }
        # Written into the pending wandb_kwargs, not onto wandb.run: wandb.init()
        # is called later, by skrl's Agent.init(), so wandb.run is still None
        # here and a live config.update() silently no-ops (measured: every window
        # key came back None on the first launch that tried it). The kwargs dict
        # is the same object the wandb_kwargs block above populated, so mutating
        # it now still reaches init.
        if not args_cli.no_wandb:
            agent_cfg["agent"].setdefault("experiment", {}).setdefault(
                "wandb_kwargs", {}).setdefault("config", {}).update(_win_cfg)
        try:
            import wandb as _wb
            if _wb.run is not None:      # sweep path: init already happened
                _wb.run.config.update(_win_cfg, allow_val_change=True)
        except Exception:
            pass  # wandb off — logging must never break a run

    if getattr(args_cli, "fix_ref_trajectories", False):
        if not hasattr(raw_env.unwrapped, "set_fix_ref_trajectories"):
            raise SystemExit("--fix_ref_trajectories requires a path-tracking env (got "
                             f"{type(raw_env.unwrapped).__name__})")
        raw_env.unwrapped.set_fix_ref_trajectories(True)

    # O6 analytic-potential critic — see --critic_analytic_potential.
    if getattr(args_cli, "critic_analytic_potential", False):
        _cb = agent_cfg.get("models", {}).get("critic")
        if isinstance(_cb, dict):
            _cb["analytic_potential"] = True

    _critic_block = agent_cfg.get("models", {}).get("critic")
    if isinstance(_critic_block, dict):
        # Same explicit-wins rule as the policy block above: an explicitly passed
        # flag overrides the yaml, an absent one leaves the yaml (or a
        # sweep-sampled value) and falls back only if nothing set it.
        _enc = args_cli.critic_encoder or args_cli.encoder
        if _enc is not None:
            _critic_block["encoder"] = _enc
        else:
            _critic_block.setdefault("encoder", "mlp")
        if args_cli.critic_encoder_stride is not None:
            _critic_block["encoder_stride"] = args_cli.critic_encoder_stride
        else:
            _critic_block.setdefault("encoder_stride", 1)
        _critic_block.setdefault("combine", args_cli.critic_combine)
        _critic_block.setdefault("embed_dim", args_cli.critic_embed_dim)

    # Wrapper order is load-bearing: StatManagerEnvWrapper must see the flat
    # tensor observations BatchedGymnasiumWrapper produces, and WandbPlotWrapper
    # sits outermost so it observes every step() regardless of caller.
    env = WandbPlotWrapper(
        StatManagerEnvWrapper(BatchedGymnasiumWrapper(raw_env)),
        total_timesteps=agent_cfg["trainer"]["timesteps"],
    )

    _annealing = normalize_agent_cfg(agent_cfg, algorithm=algorithm)
    # C2RL declares caps_* as real cfg fields and patches its own inner PPO/SAC
    # sub-agent, so its keys must stay in the dict (pop=False); skrl's Runner
    # would reject them, so the standalone route strips them. See _resolve_caps_kwargs.
    _caps = _resolve_caps_kwargs(agent_cfg, args_cli, pop=not _is_contraction)

    if _is_contraction:
        from c2rl.runners import ContractionRunner
        runner = ContractionRunner(env, agent_cfg, task_id=args_cli.task,
                                   num_envs=num_envs, is_classic=True)
    else:
        # Standalone PPO/SAC build their models from agent_cfg alone (no env
        # access), so angle_idx has to be injected into the model blocks for
        # their networks to embed it continuously — see _inject_angle_idx.
        # ContractionRunner reads it off the env itself and needs no injection.
        from c2rl.agents.skrl.runner import CLActorRunner
        _inject_angle_idx(agent_cfg, list(getattr(raw_env.unwrapped, "angle_idx", []) or []),
                          _resolve_symmetry_for_env(raw_env))
        # network_architecture: sweep-friendly override applied to both the
        # actor (models.policy) and critic (models.value) hidden layers, so a
        # PPO architecture sweep stays apples-to-apples against c3m's
        # actor_architecture (which only has a policy net to vary — C3M has no
        # critic) — see search/configs/ppo.yaml and c3m.yaml's comment on it.
        _arch = agent_cfg.pop("network_architecture", None)
        if _arch is not None:
            for _blk in ("policy", "value"):
                _net = agent_cfg.get("models", {}).get(_blk, {}).get("network")
                if isinstance(_net, list) and _net:
                    _net[0]["layers"] = list(_arch)
        runner = CLActorRunner(env, agent_cfg)

    # Contraction algorithms already namespace their own track_data() keys.
    apply_agent_patches(runner.agent, algorithm=algorithm, annealing=_annealing,
                        caps=_caps, namespace=not _is_contraction)

    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint) if _is_contraction else runner.agent.load(args_cli.checkpoint)
    runner.run()
    env.close()

    # Post-training reporting must never void a finished run. Training is done and
    # the sweep's objective (Reward/discounted_return_mean) was logged during it,
    # so a failure in the final evaluation costs some reporting detail, not the
    # trial. Left unguarded, one FileNotFoundError writing eval_results.json
    # turned every completed sweep trial into a crashed one -- and a crashed
    # trial's metric is what bayes bookkeeping ignores, so the search learned
    # nothing from runs that had actually succeeded.
    #
    # Deliberately broad: this is the last statement before exit, there is
    # nothing downstream to protect, and any exception class here is strictly
    # less important than the run that already completed. The traceback is
    # printed in full rather than swallowed.
    try:
        _evaluate_classic_path_tracking(task=args_cli.task, runner=runner, args_cli=args_cli,
                                        _is_classic=_is_classic)
    except Exception:
        import traceback
        print("[Eval] WARNING: final evaluation FAILED — training itself completed "
              "and its metrics are already logged. Traceback follows; exiting 0.",
              flush=True)
        traceback.print_exc()
    if args_cli.eval_held_out_seed is not None:
        _evaluate_classic_path_tracking(task=args_cli.task, runner=runner, args_cli=args_cli,
                                        _is_classic=_is_classic, label="HeldOut",
                                        held_out_seed=args_cli.eval_held_out_seed,
                                        held_out_trajectories=args_cli.eval_held_out_trajectories)
    finish_wandb(args_cli)


# ══════════════════════════════════════════════════════════════════════════════
# ISAAC sim route  (default)
# ══════════════════════════════════════════════════════════════════════════════
else:
    import time

    import skrl
    from packaging import version

    SKRL_VERSION = "2.0.0"
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        skrl.logger.error(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        sys.exit(1)

    if args_cli.ml_framework.startswith("torch"):
        from c2rl.agents.skrl.runner import CLActorRunner as Runner
    elif args_cli.ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner

    import c2rl.tasks  # noqa: F401

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    if args_cli.agent is None:
        agent_cfg_entry_point = f"skrl_{algorithm.replace('-', '_')}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
        # Algorithm-aware num_envs defaults: off-policy SAC-based algorithms
        # need far fewer parallel envs (large replay buffer >> many envs),
        # while on-policy PPO-based algorithms benefit from massively parallel
        # envs.  The user can always override with --num_envs.
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        elif algorithm.lower() in _SAC_LIKE_ALGOS:
            env_cfg.scene.num_envs = _DEFAULT_NUM_ENVS_SAC
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError("Distributed training is not supported on CPU.")
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        if args_cli.num_timesteps is not None:
            agent_cfg["trainer"]["timesteps"] = args_cli.num_timesteps
        agent_cfg["trainer"]["close_environment_at_exit"] = False

        # Agent-config overrides (flag name == YAML key); see _apply_agent_overrides.
        _apply_agent_overrides(agent_cfg, args_cli)

        if args_cli.ppo_activations is not None:
            models_cfg = agent_cfg.get("models", {})
            for model_type in ["policy", "value"]:
                if model_type in models_cfg:
                    for layer in models_cfg[model_type].get("network", []):
                        if "activations" in layer:
                            layer["activations"] = args_cli.ppo_activations

        if args_cli.ppo_network_arch is not None:
            arch_str = args_cli.ppo_network_arch.replace("[", "").replace("]", "")
            layers = [int(x.strip()) for x in arch_str.split(",")]
            models_cfg = agent_cfg.get("models", {})
            for model_type in ["policy", "value"]:
                if model_type in models_cfg:
                    for layer in models_cfg[model_type].get("network", []):
                        if "layers" in layer:
                            layer["layers"] = layers

        if args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        env_cfg.seed = agent_cfg["seed"]

        log_root_path = os.path.abspath(
            os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        )
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)

        # Sweep trials are throwaway: don't let hundreds of trials each write
        # ~10 full checkpoints to logs/ (see play.py:239,327 for the same idiom).
        if "WANDB_SWEEP_ID" in os.environ:
            agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

        # W&B
        _wandb_video_thread = None
        _wandb_stop_event = None
        if not args_cli.no_wandb:
            import glob as _glob
            import re as _re
            import threading as _threading

            import wandb as _wandb

            agent_cfg["agent"]["experiment"]["wandb"] = ("WANDB_SWEEP_ID" not in os.environ)
            agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})["sync_tensorboard"] = False
            install_wandb_scalar_hook()

            if args_cli.video:
                _video_dir = os.path.join(log_dir, "videos", "train")
                _uploaded_videos: set = set()
                _wandb_stop_event = _threading.Event()
                _video_metric_defined = [False]

                def _upload_pending_videos(step: int | None = None) -> None:
                    for mp4 in sorted(_glob.glob(os.path.join(_video_dir, "*.mp4"))):
                        if mp4 not in _uploaded_videos and _wandb.run is not None:
                            if not _video_metric_defined[0]:
                                # The video-watcher thread uploads asynchronously (polling every
                                # 30s) and can land after the main loop has already logged
                                # scalars at a later step — wandb rejects any step= that isn't
                                # monotonically increasing across all calls in the run. Give the
                                # video its own x-axis instead of the shared step counter, so an
                                # out-of-order upload is never rejected (https://wandb.me/define-metric).
                                _wandb.define_metric("train/video_step")
                                _wandb.define_metric("train/video", step_metric="train/video_step")
                                _video_metric_defined[0] = True
                            m = _re.search(r"step-(\d+)", os.path.basename(mp4))
                            log_step = int(m.group(1)) if m else step
                            try:
                                _wandb.log({"train/video": _wandb.Video(mp4, format="mp4"), "train/video_step": log_step})
                                _uploaded_videos.add(mp4)
                            except Exception as _e:
                                logger.warning(f"wandb video upload failed: {_e}")

                def _video_watcher() -> None:
                    while not _wandb_stop_event.is_set():
                        _upload_pending_videos()
                        _wandb_stop_event.wait(timeout=30)
                    _upload_pending_videos()

                _wandb_video_thread = _threading.Thread(target=_video_watcher, daemon=True)
                _wandb_video_thread.start()

        _wkw = agent_cfg["agent"]["experiment"].setdefault("wandb_kwargs", {})
        if args_cli.wandb_project is not None:
            _wkw["project"] = args_cli.wandb_project
        # Consistent run name: CLI override > YAML-provided name > deterministic
        # default that matches the local experiment_name (tensorboard dir), so
        # every W&B run can be correlated with its local log directory.
        _wkw["name"] = args_cli.wandb_run_name or _wkw.get("name") or agent_cfg["agent"]["experiment"]["experiment_name"]
        # See the classic branch: run.name isn't filterable, run_batch is.
        _wkw.setdefault("config", {}).update({
            "run_name": _wkw["name"],
            "run_batch": args_cli.wandb_batch or datetime.now().strftime("%Y-%m-%d"),
        })

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.log_dir = log_dir

        if hasattr(env_cfg, "vel_cmd"):
            vc = env_cfg.vel_cmd
            print(
                f"[INFO] Velocity command distribution:\n"
                f"         vx       ~ U{vc.vx_range}\n"
                f"         vy       ~ U{vc.vy_range}\n"
                f"         yaw amp  ~ U{vc.yaw_A_range} rad/s\n"
                f"         yaw freq ~ U{vc.yaw_omega_range} rad/s\n"
                f"         yaw phi  ~ U[0, 2π]"
            )

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        if args_cli.debug_vis or args_cli.video:
            # Recording a video implies you want to see the tracked-vs-target velocity
            # arrows in it, not just the raw robot — enable debug markers automatically
            # instead of requiring a separate --debug_vis flag on top of --video.
            env.unwrapped.set_debug_vis(True)

        if args_cli.video:
            video_len = args_cli.video_length if args_cli.video_length > 0 else getattr(env.unwrapped, "max_episode_length", 200)
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": video_len,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()
        _isaac_env = env  # save reference for get_physical_state() during ref-traj generation
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        from c2rl.agents.skrl.contraction_metrics import StatManagerEnvWrapper
        env = StatManagerEnvWrapper(env)
        # Raw (un-plot-wrapped) reference for _evaluate_best_model/_generate_ref_trajs:
        # both toggle skrl_env._reset_once directly to force a real sim reset, which
        # is an attribute set — WandbPlotWrapper.__getattr__ only intercepts GETs, so
        # setting it through the plot wrapper would shadow it on the wrapper instance
        # instead of the real IsaacLabWrapper, silently breaking that reset.
        _skrl_env = env
        import os as _os
        import sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from wandb_plot_wrapper import WandbPlotWrapper
        # WandbPlotWrapper must wrap the SKRL-wrapped env (flat tensor obs +
        # .state()), not the raw Isaac env — the raw env's step() returns a dict
        # obs ({"policy": ...}), which crashes WandbPlotWrapper's trajectory
        # extraction (obs[i, :3] on a dict). Order matches the classic branch above.
        env = WandbPlotWrapper(env, total_timesteps=agent_cfg["trainer"]["timesteps"])

        # angle_idx (e.g. yaw at [2] on quadruped/humanoid path-tracking) is an
        # Env attribute, not a cfg field — inject it into agent_cfg["models"] so
        # standalone PPO/SAC's _gaussian_factory/_deterministic_factory (which
        # only see per-block yaml kwargs) embed it too. ContractionRunner
        # (C3M/LQR/SDLQR/C2RL) reads it directly off the env and needs no
        # injection. _isaac_env is the pre-SkrlVecEnvWrapper raw env saved above.
        _angle_idx = list(getattr(_isaac_env.unwrapped, "angle_idx", []) or [])
        _inject_angle_idx(agent_cfg, _angle_idx, _resolve_symmetry_for_env(_isaac_env))

        _alg = agent_cfg["agent"].get("class", "").lower()
        _is_contraction = _alg in _CONTRACTION_ALGOS
        _annealing = normalize_agent_cfg(agent_cfg, algorithm=_alg)

        # C2RL reads the caps_* keys off its own cfg dataclass, so they must stay
        # in the dict there; skrl's Runner would reject them, so they are popped
        # on the standalone branch. See _resolve_caps_kwargs.
        _caps = _resolve_caps_kwargs(agent_cfg, args_cli, pop=not _is_contraction)

        if _is_contraction:
            from c2rl.runners import ContractionRunner
            # Isaac envs have no closed-form dynamics, so they always learn a
            # NeuralDynamics (pretrain + online). Forcing use_empirical_dynamics=True
            # here also makes the runner's guard reject any config that tries to
            # request analytical dynamics (use_empirical_dynamics=False) for Isaac.
            agent_cfg["agent"]["use_empirical_dynamics"] = True
            # lqr/sdlqr/cvstem-lqr own no dynamics model — they need a pretrained
            # one loaded from disk here (ContractionRunner raises with the flag
            # to pass if it's missing). See --dynamics_checkpoint.
            if args_cli.dynamics_checkpoint:
                agent_cfg["agent"]["dynamics_checkpoint"] = args_cli.dynamics_checkpoint
            runner = ContractionRunner(env, agent_cfg, is_classic=False)
        else:
            runner = Runner(env, agent_cfg)

        # Every patch no-ops on C2RL's outer agent (it owns no .policy/.scheduler/
        # .entropy_optimizer) — C2RLAgent applies them to its inner PPO/SAC
        # sub-agent itself, which is where they matter. Contraction algorithms
        # already namespace their own track_data() keys, hence namespace=False.
        apply_agent_patches(runner.agent, algorithm=_alg, annealing=_annealing,
                            caps=_caps, namespace=not _is_contraction)

        if _is_contraction and hasattr(runner.agent, "policy") and hasattr(runner.agent.policy, "cl_actor"):
            _orig_post = runner.agent.post_interaction

            def _annealed_post(*, timestep: int, timesteps: int) -> None:
                runner.agent.policy.cl_actor.anneal_stddev(timestep / max(1, timesteps))
                _orig_post(timestep=timestep, timesteps=timesteps)

            runner.agent.post_interaction = _annealed_post

        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        runner.run()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")

        if _wandb_stop_event is not None:
            _wandb_stop_event.set()
        if _wandb_video_thread is not None:
            _wandb_video_thread.join(timeout=120)

        # Best-model evaluation (CAC-dev-style: reward/AUC/contraction-rate/
        # overshoot with 95% CI) applies to PATH-Tracking envs — that's where
        # a genuine reference-trajectory tracking error is defined and where
        # C3M/LQR/SD-LQR/C2RL's contraction analysis is meaningful. It also
        # runs for VelTracking (which exposes the same get_tracking_error()
        # duck-type against a velocity command instead of a trajectory) since
        # that's used as this quality gate before ref-traj generation.
        # _evaluate_best_model no-ops with a skipped message for any env that
        # doesn't implement get_tracking_error(), so it's safe to always call.
        _evaluate_best_model(
            task=args_cli.task,
            runner=runner,
            isaac_env=_isaac_env,
            skrl_env=_skrl_env,
            env_cfg=env_cfg,
            args_cli=args_cli,
        )

        if "VelTracking" in (args_cli.task or ""):
            _generate_ref_trajs(
                task=args_cli.task,
                runner=runner,
                isaac_env=_isaac_env,
                skrl_env=_skrl_env,
                env_cfg=env_cfg,
                args_cli=args_cli,
            )

        env.close()

        finish_wandb(args_cli)

    if __name__ == "__main__":
        main()
        if not _is_classic:
            simulation_app.close()
            import os
            os._exit(0)
