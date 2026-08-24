from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import H1_CFG


@configclass
class HumanoidPathTrackingEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    decimation = 4
    episode_length_s = 40.0

    # state_dim = 50: xy_rel(2) + yaw(1) + proj_gravity_b(3) + joint_pos_rel(19)
    #                 + base_lin_vel_b(3) + base_ang_vel_b(3) + joint_vel(19)
    # ("Option A" — see quadruped_path_tracking/env.py's docstring for the
    # rationale: world SE(2) pose (xy, yaw, cyclic) + body-frame twist is the
    # minimal Markov floating-base state that also tracks the world path.
    # yaw is the only raw (wrapping) angle — angle_idx=[2].)
    # obs: x(50) + x_ref(50) + u_ref(19) = 119
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    observation_space = 119
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    action_scale = 0.25
    traj_path: str = "data/humanoid/dynamics_data.npz"
    terminate_on_fall: bool = False  # non-terminating: see quadruped_path_tracking cfg
    base_height_min = 0.50

    # uniform joint-position noise added to x_ref[0] at reset [rad]
    init_noise_scale: float = 0.05
    # uniform xy/yaw noise added to x_ref[0]'s base pose at reset — gives the
    # controller a nonzero e(0) in those dims too, same role as init_noise_scale
    init_xy_noise_scale: float = 0.1     # [m]
    init_yaw_noise_scale: float = 0.1    # [rad]
