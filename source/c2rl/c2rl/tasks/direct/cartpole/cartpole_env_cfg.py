# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env

    num_envs = 4096
    decimation = 2
    episode_length_s = 5.0
    # action: normalised cart force in [-1, 1], scaled by action_scale [N]
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
