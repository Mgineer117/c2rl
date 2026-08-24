from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class ManipulatorPathTrackingEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    decimation = 2
    episode_length_s = 15.0

    # state_dim = 21, action_dim = 7
    # obs: x(21) + x_ref(21) + u_ref(7) = 49
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = 49
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=True)

    arm_joint_names = ["panda_joint[1-7]"]
    ee_body_name = "panda_hand"

    traj_path: str = "data/manipulator/dynamics_data.npz"
