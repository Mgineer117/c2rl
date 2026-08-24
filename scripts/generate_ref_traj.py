"""Generate reference trajectories from a trained velocity-tracking policy.

The script rolls out the trained policy for many episodes, records the
physical robot state and actions at each step, and saves the results as
a compressed NumPy archive.  The path-tracking environments load these
archives to provide (x_ref, u_ref) to the contraction-tracking agent.

Usage
-----
    python scripts/generate_ref_traj.py \\
        --task    Template-Quadruped-VelTracking-v0 \\
        --checkpoint logs/skrl/quadruped_vel_tracking/.../checkpoints/agent_*.pt \\
        --robot   quadruped \\
        --num_envs 128 \\
        --num_trajs 2000 \\
        --headless

Outputs (default --out_dir data/{robot}, matching each path-tracking env's
default `traj_path`)
-------
    data/{robot}/dynamics_data.npz   — x, u, x_dot (consumed by TrajectoryBuffer)
    data/{robot}/{robot}.npz         — legacy states/actions-only archive
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--task",       type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl agent checkpoint (.pt)")
parser.add_argument("--robot",      type=str, required=True, choices=["quadruped", "humanoid", "manipulator"])
parser.add_argument("--num_envs",   type=int, default=128)
parser.add_argument("--num_trajs",  type=int, default=2000, help="Target number of episodes to record")
parser.add_argument("--out_dir",    type=str, default=None,
                     help="Output directory. Defaults to data/{robot}, matching each "
                          "path-tracking env's default traj_path — override only if "
                          "you also override traj_path in the env config.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
args.out_dir = args.out_dir or os.path.join("data", args.robot)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ------------------------------------------------------------------ #
# Everything below runs inside the Isaac Sim context
# ------------------------------------------------------------------ #

import c2rl.tasks  # noqa: F401 — registers all envs
import gymnasium as gym
import numpy as np
import torch
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils.hydra import hydra_task_config

# Map from robot name to the physical-state extractor method name
_STATE_METHOD = "get_physical_state"


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    import torch as _torch
    def _get_device():
        if _torch.cuda.is_available():
            return "cuda"
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = args.device or _get_device()
    env_cfg.scene.num_envs = args.num_trajs  # one env per trajectory → single episode collects everything
    env_cfg.sim.device = device
    agent_cfg["device"] = device
    agent_cfg["trainer"]["timesteps"] = 0    # we drive the loop manually

    # Enable initial-state randomisation for trajectory diversity
    if hasattr(env_cfg, "randomize_init"):
        env_cfg.randomize_init = True
        print(
            f"[INFO] randomize_init=True  "
            f"(pos ±{env_cfg.init_pos_range:.2f} m, "
            f"joint ±{env_cfg.init_joint_noise:.3f} rad, yaw uniform)"
        )

    # Print the velocity command distribution that was used to train the policy
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

    # Build env + wrapper
    env_raw = gym.make(args.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env_raw, ml_framework="torch")
    unwrapped = env_raw.unwrapped

    # Build runner to instantiate the trained agent with correct model/memory
    runner = Runner(env, agent_cfg)
    import logging as _logging
    _skrl_log = _logging.getLogger("skrl")
    _prev_level = _skrl_log.level
    _skrl_log.setLevel(_logging.ERROR)
    runner.agent.load(args.checkpoint)
    _skrl_log.setLevel(_prev_level)
    for model in runner.agent.models.values():
        if model is not None:
            model.eval()

    num_envs = env.num_envs  # == num_trajs
    T = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    device = next(runner.agent.policy.parameters()).device

    def _get_obs(obs_or_dict):
        return obs_or_dict["policy"] if isinstance(obs_or_dict, dict) else obs_or_dict

    obs_dict, _ = env.reset()
    obs = _get_obs(obs_dict)

    # Probe dims
    with torch.no_grad():
        actions_probe, _ = runner.agent.act(obs, None, timestep=0, timesteps=0)
    state_probe = getattr(unwrapped, _STATE_METHOD)()
    state_dim  = state_probe.shape[-1]
    action_dim = actions_probe.shape[-1]

    # Pre-allocate (num_trajs, T, dim) on GPU
    states_buf  = torch.zeros(num_envs, T, state_dim,  device=device)
    actions_buf = torch.zeros(num_envs, T, action_dim, device=device)
    ever_done   = torch.zeros(num_envs, dtype=torch.bool, device=device)

    print(f"[INFO] Collecting {num_envs} trajectories in parallel over {T} steps …")
    obs_dict, _ = env.reset()
    obs = _get_obs(obs_dict)

    for t in range(T):
        with torch.no_grad():
            actions, _ = runner.agent.act(obs, None, timestep=0, timesteps=0)
        states_buf[:, t]  = getattr(unwrapped, _STATE_METHOD)()
        actions_buf[:, t] = actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = _get_obs(obs_dict)
        ever_done |= terminated.squeeze(-1)  # only early falls, not natural time-limit truncation

    # Keep only envs that survived the full episode
    good = ~ever_done
    n_good = good.sum().item()
    print(f"[INFO] {n_good} / {num_envs} full-length trajectories (no early termination).")

    states_arr  = states_buf[good].cpu().numpy().astype(np.float32)
    actions_arr = actions_buf[good].cpu().numpy().astype(np.float32)

    # Dynamics: x_dot via 4th-order finite differences
    dt = env_cfg.sim.dt * env_cfg.decimation

    def _fd4(x: np.ndarray, h: float) -> np.ndarray:
        """4th-order finite difference along axis=1 (time). x: (N, T, D)."""
        N, T_len, D = x.shape
        d = np.empty_like(x)
        # forward (i=0)
        d[:, 0] = (-25*x[:,0] + 48*x[:,1] - 36*x[:,2] + 16*x[:,3] - 3*x[:,4]) / (12*h)
        # forward-biased (i=1)
        d[:, 1] = (-3*x[:,0] - 10*x[:,1] + 18*x[:,2] - 6*x[:,3] + x[:,4]) / (12*h)
        # central (i=2..T-3)
        d[:, 2:-2] = (x[:,:-4] - 8*x[:,1:-3] + 8*x[:,3:-1] - x[:,4:]) / (12*h)
        # backward-biased (i=T-2)
        d[:,-2] = (-x[:,-5] + 6*x[:,-4] - 18*x[:,-3] + 10*x[:,-2] + 3*x[:,-1]) / (12*h)
        # backward (i=T-1)
        d[:,-1] = (3*x[:,-5] - 16*x[:,-4] + 36*x[:,-3] - 48*x[:,-2] + 25*x[:,-1]) / (12*h)
        return d

    # angle_idx columns (e.g. yaw) wrap at ±pi in states_arr — a raw finite
    # difference across that wrap would spike x_dot by ~2*pi/dt for one sample,
    # corrupting the NeuralDynamics fit on the angle dimension. Difference an
    # Unwrapped copy instead (np.unwrap makes each angle continuous over time);
    # the saved states_arr stays wrapped (that's the physical observation), so
    # this is purely a finite-difference cleanup, not a change to x. Mirrors
    # train.py's _generate_ref_trajs.
    angle_idx = list(getattr(unwrapped, "angle_idx", []) or [])
    diff_states = states_arr.copy()
    for idx in angle_idx:
        diff_states[:, :, idx] = np.unwrap(diff_states[:, :, idx], axis=1)
    x_dot_arr = _fd4(diff_states, dt)

    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"{args.robot}.npz")
    np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
    print(f"[INFO] Saved ref trajs → {out_path}")
    print(f"       states  shape: {states_arr.shape}")
    print(f"       actions shape: {actions_arr.shape}")

    dyn_path = os.path.join(args.out_dir, "dynamics_data.npz")
    np.savez_compressed(dyn_path, x=states_arr, u=actions_arr, x_dot=x_dot_arr)
    print(f"[INFO] Saved dynamics  → {dyn_path}")
    print(f"       x      shape: {states_arr.shape}")
    print(f"       u      shape: {actions_arr.shape}")
    print(f"       x_dot  shape: {x_dot_arr.shape}")

    # Visualization: 20 random trajectories — XY positions (from states_arr relative position)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_viz  = min(20, n_good)
    chosen = np.random.choice(n_good, n_viz, replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, n_viz))

    fig, ax = plt.subplots(figsize=(8, 8))
    for k, i in enumerate(chosen):
        xy = states_arr[i, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[k], linewidth=1, alpha=0.8)
        ax.plot(xy[0,  0], xy[0,  1], "o", color=colors[k], markersize=6)
        ax.plot(xy[-1, 0], xy[-1, 1], "x", color=colors[k], markersize=8, markeredgewidth=2)
    ax.set_xlabel("x [m, relative]")
    ax.set_ylabel("y [m, relative]")
    ax.set_title("Reference trajectories — relative position  (o = start,  × = end)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    viz_path = os.path.join(args.out_dir, f"{args.robot}_viz.png")
    fig.savefig(viz_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Visualization → {viz_path}")

    env.close()


if __name__ == "__main__":
    main()
    sim_app.close()
