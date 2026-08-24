from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..common.state_guard import carry_forward_nonfinite
from ..common.vel_commands import VelCommands
from .env_cfg import ManipulatorVelTrackingEnvCfg


class ManipulatorVelTrackingEnv(DirectRLEnv):
    """Franka Panda EE-velocity-tracking environment.

    Commands: constant (vx, vy, vz) EE linear velocity + sinusoidal EE yaw rate.

    State for path-tracking export (21 dims):
        joint_pos(7) + joint_vel(7) + ee_pos_local(3)
        + ee_lin_vel(3) + ee_yaw_vel(1)
    """

    cfg: ManipulatorVelTrackingEnvCfg

    STATE_DIM = 21

    def __init__(self, cfg: ManipulatorVelTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_ids, arm_names = self._robot.find_joints(self.cfg.arm_joint_names)
        self._ee_id, _ = self._robot.find_bodies(self.cfg.ee_body_name)

        # per-joint action scale: panda_forearm (joints 5-7) is torque-limited at
        # 12 Nm vs panda_shoulder's 87 Nm, so it needs a much smaller position-delta
        # scale to avoid saturating its actuator (see env_cfg.py for the derivation).
        _forearm_joints = {"panda_joint5", "panda_joint6", "panda_joint7"}
        self._action_scale = torch.tensor(
            [
                self.cfg.action_scale_forearm if name in _forearm_joints else self.cfg.action_scale_shoulder
                for name in arm_names
            ],
            device=self.device,
        )

        self._actions = torch.zeros(self.num_envs, len(self._arm_ids), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)
        self._episode_vel_auc = torch.zeros(self.num_envs, device=self.device)
        self._episode_vel_initial = torch.zeros(self.num_envs, device=self.device)
        self._episode_vel_last = torch.zeros(self.num_envs, device=self.device)

        # divergence guard: carry the last finite physical state/reward
        # forward instead of letting a NaN/Inf physics state reach the
        # policy, value fn, or the ref-traj collector (get_physical_state()).
        self._last_valid_state: torch.Tensor | None = None
        self._last_valid_reward: torch.Tensor | None = None

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # Explicit, strongly-contrasting ground color (dark blue-teal) instead of relying on
        # GroundPlaneCfg's default grid-texture tint — the robot is light-colored, so a dark,
        # saturated (non-gray) ground reads clearly against it regardless of scene brightness.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()
        default_pos = self._robot.data.default_joint_pos[:, self._arm_ids]
        self._joint_targets = default_pos + self._action_scale * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._arm_ids)

    def get_physical_state(self) -> torch.Tensor:
        """Returns (N, 21) physical state: joint_pos(7) + joint_vel(7)
        + ee_pos_local(3) + ee_lin_vel(3) + ee_yaw_vel(1).
        """
        ee_pos = self._robot.data.body_pos_w[:, self._ee_id[0], :] - self.scene.env_origins   # (N, 3)
        ee_lin_vel = self._robot.data.body_lin_vel_w[:, self._ee_id[0], :]                     # (N, 3)
        ee_yaw_vel = self._robot.data.body_ang_vel_w[:, self._ee_id[0], 2:3]                   # (N, 1)
        state = torch.cat(
            [
                self._robot.data.joint_pos[:, self._arm_ids],
                self._robot.data.joint_vel[:, self._arm_ids],
                ee_pos,
                ee_lin_vel,
                ee_yaw_vel,
            ],
            dim=-1,
        )
        if self._last_valid_state is None:
            self._last_valid_state = torch.nan_to_num(state)
        state = carry_forward_nonfinite(state, self._last_valid_state)
        self._last_valid_state = state
        return state

    def _get_observations(self) -> dict:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)                   # (N, 8): [vx, vy, vz, yaw_rate, A, omega, sin(phase), cos(phase)]
        state = self.get_physical_state()          # (N, 21)
        obs = torch.cat([state, cmds, self._prev_actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)

        ee_lin_vel = self._robot.data.body_lin_vel_w[:, self._ee_id[0], :]      # (N, 3)
        ee_yaw_vel = self._robot.data.body_ang_vel_w[:, self._ee_id[0], 2]      # (N,)

        # velocity tracking error
        vel_err = torch.sum(torch.square(ee_lin_vel - cmds[:, :3]), dim=1)
        rew_vel = vel_err * self.cfg.rew_ee_vel

        yaw_err = torch.square(ee_yaw_vel - cmds[:, 3])
        rew_yaw = yaw_err * self.cfg.rew_ee_yaw

        rew_ar = torch.sum(torch.square(self._actions - self._prev_actions), dim=1) * self.cfg.rew_action_rate

        vel_err_vec = torch.cat([
            ee_lin_vel - cmds[:, :3],
            (ee_yaw_vel - cmds[:, 3]).unsqueeze(-1),
        ], dim=-1)
        err_norm = torch.norm(vel_err_vec, dim=-1)
        self._episode_vel_auc += err_norm
        self._episode_vel_last = err_norm.detach().clone()

        # episode_length_buf is incremented to 1 before _get_rewards runs (see
        # DirectRLEnv.step), so it is never 0 here — use <= 1 to capture the
        # first reward step. Matches quadruped_vel_tracking.
        is_first_step = self.episode_length_buf <= 1
        self._episode_vel_initial[is_first_step] = err_norm[is_first_step]

        # soft joint-limit penalty: penalise if |q - q_mid| > 0.9 * half_range
        limits = self._robot.data.soft_joint_pos_limits[:, self._arm_ids, :]
        mid = 0.5 * (limits[..., 0] + limits[..., 1])
        half_range = 0.5 * (limits[..., 1] - limits[..., 0])
        normalised = (self._robot.data.joint_pos[:, self._arm_ids] - mid) / (half_range + 1e-6)
        rew_limits = torch.sum(torch.clamp(normalised.abs() - 0.9, min=0.0) ** 2, dim=1) * self.cfg.rew_joint_limits

        total_reward = rew_vel + rew_yaw + rew_ar + rew_limits
        if self._last_valid_reward is None:
            self._last_valid_reward = torch.nan_to_num(total_reward)
        total_reward = carry_forward_nonfinite(total_reward, self._last_valid_reward)
        self._last_valid_reward = total_reward
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

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
        self._cmd.reset(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root = self._robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------ #
    # Debug visualisation: blue = command EE vel, green = current EE vel
    # Arrows shown at the EE position; 3D direction via axis-angle rotation.
    # ------------------------------------------------------------------ #

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_cmd_vel_marker"):
                _arrow = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"
                cmd_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/ManipulatorVelCmd",
                    markers={"arrow": sim_utils.UsdFileCfg(
                        usd_path=_arrow, scale=(0.3, 0.03, 0.03),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    )},
                )
                cur_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/ManipulatorVelCur",
                    markers={"arrow": sim_utils.UsdFileCfg(
                        usd_path=_arrow, scale=(0.3, 0.03, 0.03),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    )},
                )
                self._cmd_vel_marker = VisualizationMarkers(cmd_cfg)
                self._cur_vel_marker = VisualizationMarkers(cur_cfg)
            self._cmd_vel_marker.set_visibility(True)
            self._cur_vel_marker.set_visibility(True)
        else:
            if hasattr(self, "_cmd_vel_marker"):
                self._cmd_vel_marker.set_visibility(False)
                self._cur_vel_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # event stream can fire before the scene is ready or during teardown
        if not hasattr(self, "scene") or not self._robot.is_initialized:
            return
        t = self.episode_length_buf.float() * self.step_dt
        cmds = self._cmd.get(t)  # (N, 8): [vx, vy, vz, yaw, ...] in world frame

        ee_pos = self._robot.data.body_pos_w[:, self._ee_id[0], :]  # (N, 3) world
        cmd_pos = ee_pos.clone(); cmd_pos[:, 2] += 0.15
        cur_pos = ee_pos.clone(); cur_pos[:, 2] -= 0.05

        ee_vel = self._robot.data.body_lin_vel_w[:, self._ee_id[0], :]  # (N, 3)

        cmd_quat = self._world_vel_to_arrow(cmds[:, :3])
        cur_quat = self._world_vel_to_arrow(ee_vel)

        fixed_scale = torch.ones(self.num_envs, 3, device=self.device)
        proto = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._cmd_vel_marker.visualize(translations=cmd_pos, orientations=cmd_quat, scales=fixed_scale, marker_indices=proto)
        self._cur_vel_marker.visualize(translations=cur_pos, orientations=cur_quat, scales=fixed_scale, marker_indices=proto)

    def _world_vel_to_arrow(self, vel_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world-frame velocity → (N, 4) wxyz quaternion (rotate +X → vel dir)."""
        speed = vel_w.norm(dim=-1, keepdim=True)  # (N, 1)
        d = vel_w / speed.clamp(min=1e-6)          # (N, 3) unit direction

        # axis = [1,0,0] × d = [0, -dz, dy]
        axis_y = -d[:, 2]
        axis_z = d[:, 1]
        axis_norm = torch.sqrt(axis_y ** 2 + axis_z ** 2)  # (N,)

        cos_a = d[:, 0].clamp(-1.0, 1.0)
        ha = torch.acos(cos_a) * 0.5
        sin_ha = torch.sin(ha)
        safe_n = axis_norm.clamp(min=1e-6)

        qw = torch.cos(ha)
        qx = torch.zeros_like(qw)
        qy = axis_y / safe_n * sin_ha
        qz = axis_z / safe_n * sin_ha
        quat = torch.stack([qw, qx, qy, qz], dim=-1)

        # degenerate: velocity along ±X
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=vel_w.device).expand_as(quat)
        flip_z = torch.tensor([0.0, 0.0, 0.0, 1.0], device=vel_w.device).expand_as(quat)
        is_singular = (axis_norm < 1e-4).unsqueeze(-1)
        is_neg_x = (is_singular) & (cos_a < 0).unsqueeze(-1)
        quat = torch.where(is_singular & ~is_neg_x, identity, quat)
        quat = torch.where(is_neg_x, flip_z, quat)

        return quat
