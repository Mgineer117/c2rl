from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from ..common.vel_commands import VelCmdCfg


@configclass
class QuadrupedVelTrackingEnvCfg(DirectRLEnvCfg):
    # env

    num_envs = 4096
    decimation = 4
    episode_length_s = 40.0

    # Unitree Go2: 12 joints (3 per leg × 4 legs)
    # state:   base_lin_vel(3) + base_ang_vel(3) + proj_gravity(3)
    #          + joint_pos_rel(12) + joint_vel(12)  = 33
    # commands: vx,vy,vz,yaw_rate + A,omega,sin(phase),cos(phase) = 8 (see
    #           VelCommands.get() — the last 4 make the yaw-rate generator's
    #           own future Markov, not just its instantaneous value)
    # gait:    sin(gait_phase), cos(gait_phase) = 2 — the phase-clock signal the
    #           trot reward is scheduled against; Must be observable or the
    #           phase-locked gait reward is non-Markov and unlearnable.
    # obs:     state(33) + commands(8) + gait(2) + prev_actions(12) = 55
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
    observation_space = 55
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, physx=PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=1))
    # Position-delta control needs the actuator's real PD gains (stiffness=25.0,
    # damping=0.5) — zero gains produce zero torque regardless of position target.
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    # Contact sensor on the four feet — force-based contact (net force magnitude
    # > force_threshold) and per-foot air-time tracking. Robust and terrain-
    # independent, unlike a geometric foot-height proxy. track_air_time gives
    # data.last_air_time + compute_first_contact() used by the feet-air-time reward.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    # video/viewport camera: the default ViewerCfg is a *static, world-fixed*
    # camera far from any particular robot (eye=(7.5,7.5,7.5), lookat=(0,0,0)),
    # so as the robot walks away during the 10s episode it shrinks into the
    # distance and looks blurry/low-res in recorded videos even though the
    # capture resolution itself is fine. Track env 0's robot instead, with a
    # closer chase-cam framing and a higher capture resolution.
    #
    # eye/lookat give a ~38 degree downward angle (vs. the previous ~18 degrees),
    # since a shallow angle put most of the frame above the horizon — showing the
    # DomeLight's untextured "sky" fill (a flat, uniform light-gray, since it has
    # no texture_file) instead of the actual (dark) ground plane below the horizon.
    # No ground-color or lighting change can fix that if the ground isn't in frame.
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

    # action: deviation from default joint positions [rad]
    action_scale = 0.25

    # termination
    # terminate_on_fall: training uses fall termination; the post-training
    # evaluator flips this to False at runtime so episodes always run the full
    # length (metrics comparable across policies regardless of fall behavior).
    terminate_on_fall: bool = True
    target_base_height = 0.34
    base_height_min = 0.20     # [m] terminate if base drops below this
    # -0.71 ≈ -cos(45°): terminate beyond ~45° tilt. The previous -0.5 (~60°)
    # left a loophole — a robot crouched at 40-55° never terminated and sat
    # accumulating negative reward for the whole episode (observed as episode
    # returns of ~-433), fattening the advantage tails that spike PPO's KL.
    fall_grav_z_max = -0.71

    # initial-state randomisation (used by generate_ref_traj.py for trajectory diversity)
    randomize_init: bool = True
    init_pos_range: float = 0.3    # [m]  uniform x,y offset around env origin
    init_joint_noise: float = 0.05  # [rad] uniform noise on joint positions

    # reward scales — legged_gym-style recipe: body-frame exp-tracking terms
    # (sigma 0.25) dominate. Falls are punished primarily by termination (lost
    # future reward); the alive bonus keeps per-step reward positive in any
    # reasonable alive state so terminating early is never advantageous, and
    # the small flat-orientation term is smooth shaping toward upright — kept
    # mild because a large one creates reward cliffs near falls, whose
    # heavy-tailed advantages spike PPO's KL and crash the adaptive LR.
    rew_alive = 0.1
    rew_lin_vel = 2.0
    rew_yaw_rate = 0.5
    rew_flat_orientation = -0.5  # on sum(projected_gravity_b[:, :2]**2)
    rew_z_vel = -0.5
    rew_roll_pitch = -0.05
    rew_action_rate = -0.01
    rew_hip = -0.3       # penalize adduction/abduction
    rew_base_height = -10.0

    # ── trot gait (phase-clock) ──────────────────────────────────────────────
    # A per-env gait phase advances at `gait_freq` full cycles/sec. The two
    # diagonal pairs (FL+RR, FR+RL) are scheduled a half-cycle apart; each foot
    # is scheduled stance for `gait_duty` of its cycle and swing otherwise. The
    # reward matches actual (force-based) contact against this schedule, so —
    # unlike a diagonal-symmetry penalty — standing and pronking are not
    # zero-penalty: their swing-scheduled feet mismatch.
    #
    # gait_duty > 0.5 makes the two diagonals' stance windows overlap, giving a
    # double-support ("all four allowed down") buffer around each diagonal swap
    # — what makes this a walk, not an instantaneous-swap run. At duty=0.6 that
    # buffer is 20% of the cycle (two ~0.1 s windows at 1 Hz) with no flight
    # phase. The cost: standing's schedule-match rises from 0.5 to ~0.6 (a small
    # positive score), but forward-velocity tracking already makes standing
    # non-viable, and a true trot still scores strictly higher.
    #
    # step_dt = sim.dt * decimation = 0.02 s, so gait_freq=1.0 → one cycle / 50
    # steps (1.0 s), ~1 footfall/sec per foot — a walking (not running) cadence.
    gait_freq = 1.0      # [Hz] gait cycles per second
    gait_duty = 0.8      # stance fraction per foot (>0.5 → double-support buffer)
    rew_gait = 1.0       # scale on the centered [-1, 1] schedule-match score
    # Half-width [rad] of the smoothstep ramp at each stance↔swing edge. The
    # desired-contact schedule ramps over this band instead of stepping, so the
    # reward has no cliff at the swap — this is the de-jitter knob. ~0.4 rad ≈
    # 0.064 s per side at 1 Hz; larger = smoother/softer gait, smaller = crisper.
    gait_transition_band = 0.2
