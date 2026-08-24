from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import H1_CFG

from ..common.vel_commands import VelCmdCfg


@configclass
class HumanoidVelTrackingEnvCfg(DirectRLEnvCfg):
    # env

    num_envs = 4096
    decimation = 4
    episode_length_s = 40.0

    # Unitree H1: 19 joints
    #   legs:  2 hip_yaw + 2 hip_roll + 2 hip_pitch + 2 knee + 1 torso = 9
    #   feet:  2 ankle = 2
    #   arms:  2 shoulder_pitch + 2 shoulder_roll + 2 shoulder_yaw + 2 elbow = 8
    # state (path-tracking export): proj_gravity(3) + joint_pos_rel(19) + joint_vel(19) = 41
    # commands: vx,vy,vz,yaw_rate + A,omega,sin(phase),cos(phase) = 8 (see
    #           VelCommands.get() — the last 4 make the yaw-rate generator's
    #           own future Markov, not just its instantaneous value)
    # obs:     full_state(47) + commands(8) + prev_actions(19) = 74
    #   (full_state still includes lin_vel_b+ang_vel_b for the locomotion policy to use)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(19,), dtype=np.float32)
    observation_space = 74
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    # Position-delta control needs the actuators' real PD gains (H1_CFG's own
    # per-joint stiffness/damping in the "legs" [incl. torso], "feet", "arms"
    # groups) — zero gains produce zero torque regardless of position target.
    robot_cfg: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True)

    # video/viewport camera: chase-cam tracking env 0's robot (see
    # quadruped_vel_tracking's env_cfg.py for the full rationale — the
    # default static ViewerCfg shrinks into the distance as the robot walks).
    viewer: ViewerCfg = ViewerCfg(
        eye=(-5.0, 8.66, 7.5),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        resolution=(1920, 1080),
    )

    # velocity commands: forward speed along the *current heading* + a yaw rate
    # that is, per episode, either held at 0 (omega=0, straight-line heading) or
    # sinusoidal (omega = 2*pi/T, one full cycle per episode), chosen 50/50 via
    # yaw_omega_binary. vy/vz are pinned to 0 so the linear command is purely
    # along heading. With phase fixed at 0, the sinusoidal case starts and ends
    # each episode at zero yaw rate, so the robot weaves left-then-right (an S)
    # and returns to its initial heading over exactly one cycle.
    vel_cmd: VelCmdCfg = VelCmdCfg(
        vx_range=(0.5, 1.5),        # forward speed [m/s] — sampled
        vy_range=(0.0, 0.0),        # no lateral component: velocity is along heading
        vz_range=(0.0, 0.0),
        yaw_A_range=(0.0, 0.5),     # yaw-rate amplitude [rad/s] — sampled
        yaw_omega_range=(0.0, 2 * math.pi / episode_length_s),  # binary: constant vs. one cycle/episode
        yaw_omega_binary=True,
        yaw_phase_range=(0.0, 0.0),  # start each episode at zero yaw rate
    )

    # action: deviation from default joint positions [rad] — matches quadruped_vel_tracking's convention
    action_scale = 0.25

    # termination
    # terminate_on_fall: training uses fall termination; the post-training
    # evaluator flips this to False at runtime so episodes always run the full
    # length (metrics comparable across policies regardless of fall behavior).
    terminate_on_fall: bool = True
    base_height_min = 0.50    # [m] terminate if base drops below this
    # -0.71 ≈ -cos(45°): terminate beyond ~45° tilt (see quadruped_vel_tracking
    # for the rationale — a shallower cutoff lets the robot sit crouched at
    # 40-55° accumulating negative reward for the whole episode).
    fall_grav_z_max = -0.71

    # initial-state randomisation (used by generate_ref_traj.py for trajectory diversity)
    randomize_init: bool = True
    init_pos_range: float = 0.3    # [m]  uniform x,y offset around env origin
    init_joint_noise: float = 0.05  # [rad] uniform noise on joint positions

    # reward scales — legged_gym-style recipe: body-frame exp-tracking terms
    # dominate; the alive bonus keeps per-step reward positive in any
    # reasonable alive state so terminating early is never advantageous.
    rew_alive = 1.0
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_flat_orientation = -1.0  # on sum(projected_gravity_b[:, :2]**2)
    rew_z_vel = -0.5
    rew_roll_pitch = -0.05
    rew_torque = -1e-5
    rew_action_rate = -0.01
