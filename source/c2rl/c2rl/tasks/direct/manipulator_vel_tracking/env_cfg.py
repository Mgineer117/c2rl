from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ..common.vel_commands import VelCmdCfg


@configclass
class ManipulatorVelTrackingEnvCfg(DirectRLEnvCfg):
    num_envs = 4096
    # env — EE Cartesian velocity tracking task
    decimation = 2
    episode_length_s = 15.0

    # Franka Panda: 7 arm joints (finger joints excluded from control)
    # state:  joint_pos(7) + joint_vel(7) + ee_pos_local(3)
    #         + ee_lin_vel(3) + ee_yaw_vel(1)  = 21
    # commands: vx,vy,vz,yaw_rate + A,omega,sin(phase),cos(phase) = 8 (see
    #           VelCommands.get() — the last 4 make the yaw-rate generator's
    #           own future Markov, not just its instantaneous value)
    # obs:    state(21) + commands(8) + prev_actions(7) = 36
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = 36
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=True)

    arm_joint_names = ["panda_joint[1-7]"]
    ee_body_name = "panda_hand"

    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(-0.3, 0.3),   # EE linear velocity [m/s]
        vy_range=(-0.3, 0.3),
        vz_range=(-0.2, 0.2),
        yaw_A_range=(0.1, 0.5),  # EE yaw rate [rad/s]
        yaw_omega_range=(0.3, 1.5),
    )

    # action: deviation from default joint positions [rad] — matches quadruped/humanoid_vel_tracking's convention
    # Both groups run at FRANKA_PANDA_HIGH_PD_CFG's stiffness=400.0, but effort_limit_sim
    # differs sharply per group (87 Nm vs 12 Nm), so a single action_scale=0.25 (needing
    # 100 Nm at full-scale action) saturates both: panda_shoulder by 15%, panda_forearm by
    # 8.3x. Scaled below each group's own limit/stiffness ratio, with ~15-20% margin left
    # for the PD law's damping*velocity_error term (not captured by this static torque estimate).
    action_scale_shoulder = 0.20   # panda_joint[1-4]: 400*0.20=80 Nm vs 87 Nm limit
    action_scale_forearm = 0.025   # panda_joint[5-7]: 400*0.025=10 Nm vs 12 Nm limit

    # reward scales (tentative)
    rew_ee_vel = -1.0       # -||ee_vel - cmd_vel||^2
    rew_ee_yaw = -0.5       # -||ee_yaw_vel - cmd_yaw||^2
    rew_action_rate = -0.01
    rew_joint_limits = -0.1  # penalty for approaching joint limits
