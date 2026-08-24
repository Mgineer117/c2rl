from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from ..common.state_guard import carry_forward_nonfinite
from ..common.vel_commands import VelCommands
from .env_cfg import QuadrupedVelTrackingEnvCfg


class QuadrupedVelTrackingEnv(DirectRLEnv):
    """Unitree Go2 velocity-tracking environment.

    Commands: constant body-frame (vx, vy) + sinusoidal yaw rate.
    State for path-tracking export (matches quadruped_path_tracking's Option A
    layout exactly, so trajectories recorded here are directly usable as
    quadruped_path_tracking's reference data):
        xy_rel(2) + yaw(1) + proj_gravity_b(3) + joint_pos_rel(12)
        + base_lin_vel_b(3) + base_ang_vel_b(3) + joint_vel(12)  → 36 dims
    """

    cfg: QuadrupedVelTrackingEnvCfg

    STATE_DIM = 36   # dims recorded as reference state for path tracking
    angle_idx = [2]  # yaw — matches quadruped_path_tracking's angle_idx; read by
                     # _generate_ref_trajs (train.py) to unwrap before finite-differencing

    def __init__(self, cfg: QuadrupedVelTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, _ = self._robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
        self._hip_ids, _ = self._robot.find_joints([".*_hip_joint"])
        # Feet ordering/contact come from the ContactSensor (force-based), not the
        # articulation — see _setup_scene. find_bodies indexes into the sensor's
        # tracked bodies, so contact/air-time columns line up with _feet_names.
        self._feet_ids, self._feet_names = self._contact_sensor.find_bodies(".*_foot")

        # Per-foot indices for the trot schedule. Guarded so a robot whose feet
        # aren't named FL/FR/RL/RR fails loudly at init instead of StopIteration
        # deep in a generator (or, worse, silently mis-indexing the gait reward).
        self._fl_idx = self._foot_index("FL")
        self._fr_idx = self._foot_index("FR")
        self._rl_idx = self._foot_index("RL")
        self._rr_idx = self._foot_index("RR")

        self._actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._prev_joint_vel = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        # Gait phase (rad) per env — advanced in _pre_physics_step, observed as
        # (sin, cos) so the phase-clock trot reward is Markov/learnable.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)
        self._episode_vel_auc = torch.zeros(self.num_envs, device=self.device)
        self._episode_vel_initial = torch.zeros(self.num_envs, device=self.device)
        self._episode_vel_last = torch.zeros(self.num_envs, device=self.device)

        # divergence guard: carry the last finite obs/reward forward instead
        # of letting a NaN/Inf physics state reach the policy or value fn.
        self._last_valid_obs: torch.Tensor | None = None
        self._last_valid_reward: torch.Tensor | None = None

    def _foot_index(self, key: str) -> int:
        """Position of the foot whose body name contains ``key`` within _feet_names."""
        for i, n in enumerate(self._feet_names):
            if key in n:
                return i
        raise ValueError(
            f"[QuadrupedVelTracking] no foot body matching '{key}' in {self._feet_names}; "
            "the trot gait reward assumes FL/FR/RL/RR foot naming."
        )

    def _stance_prob(self, foot_phase: torch.Tensor) -> torch.Tensor:
        """Smooth desired-stance probability in [0, 1] for a foot at ``foot_phase`` (rad).

        Trapezoidal window: ~1 across the stance arc [0, 2π·duty), ~0 across the
        swing arc, with smoothstep ramps of half-width ``gait_transition_band``
        at each edge. Ramping the schedule (instead of a hard step) removes the
        reward cliff at the stance↔swing swap that makes the policy snap feet
        up/down — the main source of clocked-gait leg jitter.
        """
        two_pi = 2.0 * math.pi
        half = math.pi * self.cfg.gait_duty       # half-width of the stance arc
        band = self.cfg.gait_transition_band
        # wrapped signed distance from the stance-window center (φ = half), in [-π, π]
        delta = torch.remainder(foot_phase - half + math.pi, two_pi) - math.pi
        # 0 inside (|δ| ≤ half−band) → prob 1;  1 outside (|δ| ≥ half+band) → prob 0
        x = ((delta.abs() - (half - band)) / (2.0 * band)).clamp(0.0, 1.0)
        return 1.0 - x * x * (3.0 - 2.0 * x)      # smoothstep

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # Standard Isaac Sim ground plane (grey grid texture)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # Distant light gives directional sunlight with shadows — the standard
        # Isaac Sim look. DomeLightCfg was washing everything to flat white.
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()
        default_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        self._joint_targets = default_pos + self.cfg.action_scale * self._actions
        # Advance the gait phase once per control step. Kept in [0, 2π) so
        # (sin, cos) in the observation are the sole Markov phase signal; the
        # same buffer drives this step's reward and this step's observation.
        dphi = 2.0 * math.pi * self.cfg.gait_freq * self.step_dt
        self._gait_phase = torch.remainder(self._gait_phase + dphi, 2.0 * math.pi)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    # ------------------------------------------------------------------ #
    # State extraction — also called by generate_ref_traj.py
    # ------------------------------------------------------------------ #

    def get_physical_state(self) -> torch.Tensor:
        """Returns (N, 36) physical state without commands or actions.

        Layout: [xy_rel(2), yaw(1), proj_gravity_b(3), joint_pos_rel(12),
        base_lin_vel_b(3), base_ang_vel_b(3), joint_vel(12)] — identical to
        quadruped_path_tracking's Option A state (see that env's docstring),
        so recordings from here can be replayed as path-tracking references.
        """
        xy_rel = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return torch.cat(
            [
                xy_rel,
                yaw.unsqueeze(-1),
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.joint_vel[:, self._joint_ids],
            ],
            dim=-1,
        )

    def _get_observations(self) -> dict:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)  # (N, 8): [vx, vy, vz, yaw_rate, A, omega, sin(phase), cos(phase)]
        # Full 33-dim state for the locomotion policy (needs velocities to track commands)
        full_state = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.joint_vel[:, self._joint_ids],
            ],
            dim=-1,
        )
        gait = torch.stack([torch.sin(self._gait_phase), torch.cos(self._gait_phase)], dim=-1)
        obs = torch.cat([full_state, cmds, gait, self._prev_actions], dim=-1)
        if self._last_valid_obs is None:
            self._last_valid_obs = torch.nan_to_num(obs)
        obs = carry_forward_nonfinite(obs, self._last_valid_obs)
        self._last_valid_obs = obs
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)

        # linear velocity tracking (xy) in *body* frame. The observation only
        # contains body-frame velocity and yaw-invariant projected gravity —
        # world-frame tracking would require the unobserved yaw, making the
        # task partially observable under randomize_init.
        lin_err = torch.sum(
            torch.square(cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        rew_lin = torch.exp(-lin_err / 0.25) * self.cfg.rew_lin_vel

        # yaw rate tracking
        yaw_err = torch.square(cmds[:, 3] - self._robot.data.root_ang_vel_b[:, 2])
        rew_yaw = torch.exp(-yaw_err / 0.25) * self.cfg.rew_yaw_rate

        vel_err_vec = cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2]
        err_norm = torch.norm(vel_err_vec, dim=-1)
        self._episode_vel_auc += err_norm
        self._episode_vel_last = err_norm.detach().clone()

        is_first_step = self.episode_length_buf <= 1
        self._episode_vel_initial[is_first_step] = err_norm[is_first_step]

        # flat-orientation penalty: projected gravity xy is ~0 upright, ~1 on
        # the side/back. With no fall termination, this is what makes lying
        # down strictly worse than any upright behavior.
        rew_flat = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) * self.cfg.rew_flat_orientation

        rew_z = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * self.cfg.rew_z_vel
        rew_rp = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.rew_roll_pitch
        rew_act_rate = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate

        rew_height = torch.square(self._robot.data.root_pos_w[:, 2] - self.cfg.target_base_height) * self.cfg.rew_base_height

        # Force-based foot contact + air time from the ContactSensor (robust,
        # terrain-independent — see env_cfg.contact_sensor).
        net_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids]  # (N, 4, 3)
        contact = net_forces.norm(dim=-1) > self.cfg.contact_sensor.force_threshold  # (N, 4) bool
        cmd_norm = torch.norm(cmds[:, :2], dim=1)
        moving = (cmd_norm > 0.1).float()

        # Phase-clock trot reward (smoothed). Diagonal pairs (FL+RR, FR+RL) are
        # scheduled a half-cycle apart; each foot's desired-stance probability
        # d(φ) ∈ [0,1] ramps smoothly across the stance↔swing swap (see
        # _stance_prob), so the reward has no cliff there — this is what de-jitters
        # the gait. Reward = mean over feet of agreement 1−|contact−d|, centered
        # to [-1, 1]. Standing (all-stance) and pronking (all-together) both
        # disagree with their swing-scheduled feet, so neither is zero-penalty.
        stance_A = self._stance_prob(self._gait_phase)            # FL, RR
        stance_B = self._stance_prob(self._gait_phase + math.pi)  # FR, RL
        d = torch.zeros_like(contact, dtype=torch.float32)
        d[:, self._fl_idx] = stance_A
        d[:, self._rr_idx] = stance_A
        d[:, self._fr_idx] = stance_B
        d[:, self._rl_idx] = stance_B
        c = contact.float()
        match = (1.0 - (c - d).abs()).mean(dim=1)                 # [0, 1]
        # When standing is commanded, a trot schedule is wrong — reward full
        # stance instead so the robot isn't pushed to step in place.
        stand_score = c.mean(dim=1)                               # 1.0 iff all four down
        gait_score = torch.where(moving > 0, match, stand_score)
        rew_gait = (2.0 * gait_score - 1.0) * self.cfg.rew_gait

        hip_pos = self._robot.data.joint_pos[:, self._hip_ids] - self._robot.data.default_joint_pos[:, self._hip_ids]
        rew_hip = torch.sum(torch.square(hip_pos), dim=1) * self.cfg.rew_hip

        total_reward = (self.cfg.rew_alive + rew_lin + rew_yaw + rew_flat + rew_z + rew_rp +
                        rew_height + rew_gait + rew_hip + rew_act_rate)

        if not hasattr(self, "_episode_discounted_returns"):
            self._episode_discounted_returns = torch.zeros(self.num_envs, device=self.device)
            self._current_discounts = torch.ones(self.num_envs, device=self.device)
            self._episode_undiscounted_returns = torch.zeros(self.num_envs, device=self.device)
            self._episode_lengths_custom = torch.zeros(self.num_envs, device=self.device)

        self._episode_discounted_returns += self._current_discounts * total_reward
        self._episode_undiscounted_returns += total_reward
        self._episode_lengths_custom += 1.0
        self._current_discounts *= 0.999  # matches agent discount_factor (PPO/SAC = 0.999)

        if self._last_valid_reward is None:
            self._last_valid_reward = torch.nan_to_num(total_reward)
        total_reward = carry_forward_nonfinite(total_reward, self._last_valid_reward)
        self._last_valid_reward = total_reward
        return total_reward

    def get_tracking_error(self) -> torch.Tensor:
        """Current velocity-tracking error norm per env, (N,).

        Same integrand as the Episode/auc metric: body-frame xy velocity error
        against the commanded velocity. Used by the post-training evaluator to
        fit the exponential contraction envelope.
        """
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)
        return torch.norm(cmds[:, :2] - self._robot.data.root_lin_vel_b[:, :2], dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Fall = base dropped too low or body tilted past the limit. projected
        # gravity z is -1 upright and rises toward 0/+1 as the base tilts, so
        # ``> fall_grav_z_max`` (default -0.5) fires at ~>60 deg from upright —
        # catches side/back falls that a height check alone would miss.
        too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.base_height_min
        tilted = self._robot.data.projected_gravity_b[:, 2] > self.cfg.fall_grav_z_max
        fell = too_low | tilted
        if not getattr(self.cfg, "terminate_on_fall", True):
            fell = torch.zeros_like(time_out)
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        auc_vals = self._episode_vel_auc[env_ids]
        if hasattr(self, "_episode_discounted_returns"):
            disc_returns = self._episode_discounted_returns[env_ids]
            undisc_returns = self._episode_undiscounted_returns[env_ids]
            lengths = self._episode_lengths_custom[env_ids]
            if (auc_vals > 0).any():
                self.extras.setdefault("log", {})
                mask = auc_vals > 0
                init_costs = self._episode_vel_initial[env_ids]
                init_cost = torch.clamp(init_costs[mask], min=1e-6)
                e_T = self._episode_vel_last[env_ids][mask]
                auc_trapz = auc_vals[mask] - 0.5 * init_costs[mask] - 0.5 * e_T
                self.extras["log"]["Episode/auc"] = (auc_trapz / init_cost * self.step_dt).mean()
                self.extras["log"]["Reward/discounted_return"] = disc_returns[mask].mean()
                self.extras["log"]["Reward/avg_reward_per_step"] = (undisc_returns[mask] / lengths[mask]).mean()
                # Same "Reward/total_reward_*" keys, and same per-episode-reset
                # cadence as the Stability tab, that C3M's eval loop emits via
                # track_reward_summary — skrl's own "Reward / Total reward
                # (mean)" tracker uses a different key name and can't be unified
                # with it, so it was leaving total_reward_mean written only once
                # (by the post-training evaluator) instead of throughout training.
                from c2rl.agents.skrl.contraction_metrics import reward_log_dict, reward_summary
                self.extras["log"].update(
                    reward_log_dict(reward_summary(undisc_returns, mask), self.device)
                )

            self._episode_discounted_returns[env_ids] = 0.0
            self._episode_undiscounted_returns[env_ids] = 0.0
            self._episode_lengths_custom[env_ids] = 0.0
            self._current_discounts[env_ids] = 1.0

        self._episode_vel_auc[env_ids] = 0.0
        self._episode_vel_initial[env_ids] = 0.0
        self._episode_vel_last[env_ids] = 0.0

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0
        self._gait_phase[env_ids] = 0.0
        self._cmd.reset(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]

        if self.cfg.randomize_init:
            n = len(env_ids)
            # Random yaw: sample θ ∈ [0, 2π], build wxyz quaternion around Z
            theta = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * math.pi)
            cos_h, sin_h = torch.cos(theta * 0.5), torch.sin(theta * 0.5)
            root[:, 3:7] = torch.stack(
                [cos_h, torch.zeros_like(cos_h), torch.zeros_like(cos_h), sin_h], dim=-1
            )
            # Random x,y offset (spread robots across scene)
            root[:, :2] += torch.empty(n, 2, device=self.device).uniform_(
                -self.cfg.init_pos_range, self.cfg.init_pos_range
            )
            # Joint noise: perturb around default pose for trajectory diversity
            joint_pos[:, self._joint_ids] += torch.empty(
                n, len(self._joint_ids), device=self.device
            ).uniform_(-self.cfg.init_joint_noise, self.cfg.init_joint_noise)

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------ #
    # Debug visualisation: blue = command vel, green = current vel
    #
    # Arrows are built from two 3D primitives (cylinder shaft + cone head,
    # both axis="X") rather than the flat arrow_x.usd mesh — a flat mesh
    # looks like a thin line (direction ambiguous) from many camera angles,
    # while a cylinder+cone is a real 3D solid, recognizable from any angle.
    # ------------------------------------------------------------------ #

    _ARROW_SHAFT_LEN = 0.4
    _ARROW_SHAFT_RADIUS = 0.02
    _ARROW_HEAD_LEN = 0.2
    _ARROW_HEAD_RADIUS = 0.05

    def _make_arrow_markers_cfg(self, prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        return VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "shaft": sim_utils.CylinderCfg(
                    radius=self._ARROW_SHAFT_RADIUS, height=self._ARROW_SHAFT_LEN,
                    axis="X", visual_material=material,
                ),
                "head": sim_utils.ConeCfg(
                    radius=self._ARROW_HEAD_RADIUS, height=self._ARROW_HEAD_LEN,
                    axis="X", visual_material=material,
                ),
            },
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_cmd_vel_marker"):
                cmd_cfg = self._make_arrow_markers_cfg("/Visuals/QuadrupedVelCmd", (0.0, 0.0, 1.0))
                cur_cfg = self._make_arrow_markers_cfg("/Visuals/QuadrupedVelCur", (0.0, 1.0, 0.0))
                # Yellow yaw-rate arrow: tangential to heading, pointing the way the
                # nose is commanded to swing (left = ccw / positive yaw rate).
                yaw_cfg = self._make_arrow_markers_cfg("/Visuals/QuadrupedYawCmd", (1.0, 0.9, 0.0))
                self._cmd_vel_marker = VisualizationMarkers(cmd_cfg)
                self._cur_vel_marker = VisualizationMarkers(cur_cfg)
                self._yaw_cmd_marker = VisualizationMarkers(yaw_cfg)
            self._cmd_vel_marker.set_visibility(True)
            self._cur_vel_marker.set_visibility(True)
            self._yaw_cmd_marker.set_visibility(True)
        else:
            if hasattr(self, "_cmd_vel_marker"):
                self._cmd_vel_marker.set_visibility(False)
                self._cur_vel_marker.set_visibility(False)
                self._yaw_cmd_marker.set_visibility(False)

    def _arrow_parts(self, base_pos: torch.Tensor, quat: torch.Tensor, scale_len: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (translations, orientations, marker_indices) for a shaft+head arrow at base_pos, pointing along quat.

        Cylinder/cone primitives are centered at their own local origin, so
        each part is offset along local +X by half its own length plus
        whatever precedes it — this reproduces the old arrow_x.usd mesh's
        convention of "origin at the tail, extending outward" for the
        combined shaft+head, with the cone's base flush against the shaft's tip.
        """
        n = base_pos.shape[0]
        if scale_len is None:
            scale_len = torch.ones(n, device=self.device)

        s = scale_len.unsqueeze(-1)
        shaft_offset = torch.tensor(
            [self._ARROW_SHAFT_LEN / 2, 0.0, 0.0], device=self.device
        ).expand(n, -1) * s
        head_offset = torch.tensor(
            [self._ARROW_SHAFT_LEN, 0.0, 0.0], device=self.device
        ).expand(n, -1) * s + torch.tensor(
            [self._ARROW_HEAD_LEN / 2, 0.0, 0.0], device=self.device
        ).expand(n, -1)

        shaft_pos = base_pos + quat_apply(quat, shaft_offset)
        head_pos = base_pos + quat_apply(quat, head_offset)

        translations = torch.cat([shaft_pos, head_pos], dim=0)
        orientations = torch.cat([quat, quat], dim=0)
        marker_indices = torch.cat([
            torch.zeros(n, dtype=torch.int32, device=self.device),  # 0 -> "shaft"
            torch.ones(n, dtype=torch.int32, device=self.device),   # 1 -> "head"
        ], dim=0)
        return translations, orientations, marker_indices

    def _debug_vis_callback(self, event):
        # event stream can fire before the scene is ready or during teardown
        if not hasattr(self, "scene") or not self._robot.is_initialized:
            return
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)  # (N, 8): [vx_b, vy_b, vz, yaw, ...]

        cmd_pos = self._robot.data.root_pos_w.clone()
        cmd_pos[:, 2] += 0.5  # Same height
        cur_pos = self._robot.data.root_pos_w.clone()
        cur_pos[:, 2] += 0.5  # Same height

        # commands are body-frame: rotate into world (yaw only) for display
        w, x, y, z = self._robot.data.root_quat_w.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cmd_xy_w = torch.stack(
            [cos_y * cmds[:, 0] - sin_y * cmds[:, 1],
             sin_y * cmds[:, 0] + cos_y * cmds[:, 1]], dim=-1
        )
        cmd_quat = self._vel_world_xy_to_arrow(cmd_xy_w)
        cur_quat = self._vel_world_xy_to_arrow(self._robot.data.root_lin_vel_w[:, :2])

        cmd_mag = torch.clamp(torch.norm(cmd_xy_w, dim=-1), min=0.01)
        cur_mag = torch.clamp(torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=-1), min=0.01)

        cmd_translations, cmd_orientations, cmd_indices = self._arrow_parts(cmd_pos, cmd_quat, cmd_mag)
        cur_translations, cur_orientations, cur_indices = self._arrow_parts(cur_pos, cur_quat, cur_mag)

        cmd_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        cmd_scale[:self.num_envs, 0] = cmd_mag  # scale shaft length
        cmd_scale[:, 1:] = 1.5  # Make command arrow 50% thicker

        cur_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        cur_scale[:self.num_envs, 0] = cur_mag  # scale shaft length
        cur_scale[:, 1:] = 0.7  # Make current arrow thinner

        self._cmd_vel_marker.visualize(
            translations=cmd_translations, orientations=cmd_orientations, scales=cmd_scale, marker_indices=cmd_indices
        )
        self._cur_vel_marker.visualize(
            translations=cur_translations, orientations=cur_orientations, scales=cur_scale, marker_indices=cur_indices
        )

        # ── Yaw-rate arrow (yellow) ──────────────────────────────────────── #
        # Commanded yaw rate = cmds[:, 3]. Draw a horizontal arrow tangential to
        # the heading, showing which way the nose is being commanded to swing:
        # for a point on the +x (heading) axis, the yaw-induced velocity is
        # yaw_rate · (ẑ × x̂) = yaw_rate · ŷ_body, i.e. world (-sin_y, cos_y)
        # scaled (and sign-flipped) by the yaw rate. Anchored at the nose and
        # raised above the velocity arrows so the two don't overlap.
        yaw_rate = cmds[:, 3]
        yaw_vec_w = torch.stack([-sin_y * yaw_rate, cos_y * yaw_rate], dim=-1)
        yaw_quat = self._vel_world_xy_to_arrow(yaw_vec_w)
        # visual gain so a small rad/s reads clearly; min keeps a stub at the
        # zero-crossings of the sine so the marker never fully disappears.
        yaw_mag = torch.clamp(yaw_rate.abs() * 2.0, min=0.05)
        nose = self._robot.data.root_pos_w.clone()
        nose[:, 0] += 0.3 * cos_y  # 0.3 m ahead along heading
        nose[:, 1] += 0.3 * sin_y
        nose[:, 2] += 0.7  # above the velocity arrows
        yaw_translations, yaw_orientations, yaw_indices = self._arrow_parts(nose, yaw_quat, yaw_mag)
        yaw_scale = torch.ones(2 * self.num_envs, 3, device=self.device)
        yaw_scale[:self.num_envs, 0] = yaw_mag
        yaw_scale[:, 1:] = 1.2
        self._yaw_cmd_marker.visualize(
            translations=yaw_translations, orientations=yaw_orientations, scales=yaw_scale, marker_indices=yaw_indices
        )

    def _vel_world_xy_to_arrow(self, vel_world_xy: torch.Tensor) -> torch.Tensor:
        """World-frame XY velocity → world-frame arrow quaternion (wxyz)."""
        angle = torch.atan2(vel_world_xy[:, 1], vel_world_xy[:, 0])
        ha = angle * 0.5
        return torch.stack([torch.cos(ha), torch.zeros_like(ha), torch.zeros_like(ha), torch.sin(ha)], dim=-1)
