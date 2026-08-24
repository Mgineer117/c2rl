from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

from ..common.path_tracking_base import PathTrackingBase
from .env_cfg import QuadrupedPathTrackingEnvCfg


class QuadrupedPathTrackingEnv(PathTrackingBase):
    """Unitree Go2 path-tracking environment.

    obs  = [x_current(36), x_ref(36), u_ref(12)] = 84
    reward = -||x_current - x_ref||^2  (yaw dim shortest-angle wrapped, see
             PathTrackingBase._get_rewards)

    x layout: xy_rel_to_origin(2) + yaw(1) + proj_gravity_b(3)
              + joint_pos_rel(12) + base_lin_vel_b(3) + base_ang_vel_b(3)
              + joint_vel(12)
    u layout: joint position deviation targets (12)

    "Option A": world SE(2) pose (xy, yaw) + body-frame twist — xy/yaw are
    cyclic w.r.t. the dynamics (never appear in f(x)/B(x), only their
    derivatives do via base_lin/ang_vel_b), so this is the minimal Markov
    floating-base state that also tracks the world path. yaw is the only raw
    (wrapping) angle — angle_idx=[2] — every network embeds it continuously.
    """

    cfg: QuadrupedPathTrackingEnvCfg
    angle_idx = [2]

    def __init__(self, cfg: QuadrupedPathTrackingEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, _ = self._robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone()
        default_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        self._joint_targets = default_pos + self.cfg.action_scale * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    @property
    def state_names(self) -> tuple[str, ...]:
        """Matches _get_physical_state's concatenation order exactly.

        xy_rel and yaw are the only world-frame quantities; projected gravity,
        the joint states and both root velocities are body frame, hence already
        yaw invariant -- which is what makes the full SE(2) quotient valid here.
        """
        n_j = len(self._joint_ids)
        return (
            ("pos_x", "pos_y", "yaw")
            + tuple(f"proj_grav_{a}" for a in "xyz")
            + tuple(f"joint_pos_{i}" for i in range(n_j))
            + ("vel_x_b", "vel_y_b", "vel_z_b")
            + ("ang_vel_x_b", "ang_vel_y_b", "ang_vel_z_b")
            + tuple(f"joint_vel_{i}" for i in range(n_j))
        )

    def _get_physical_state(self) -> torch.Tensor:
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = self._robot.data.root_pos_w[:, 2] < self.cfg.base_height_min
        if not getattr(self.cfg, "terminate_on_fall", True):
            fell = torch.zeros_like(time_out)
        # Non-finite/diverged states are handled by the carry-forward guard in
        # _sanitize_state, not by termination — see PathTrackingBase.
        terminated = fell
        return terminated, time_out

    def _set_robot_state_from_ref(self, env_ids: torch.Tensor, x_ref_init: torch.Tensor) -> None:
        # x layout: [xy_rel(0:2), yaw(2), gravity(3:6), joint_pos_rel(6:18),
        #            base_lin_vel_b(18:21), base_ang_vel_b(21:24), joint_vel(24:36)]
        xy_ref = x_ref_init[:, 0:2]
        yaw_ref = x_ref_init[:, 2]
        joint_pos_rel = x_ref_init[:, 6:18]
        base_lin_vel_ref = x_ref_init[:, 18:21]
        base_ang_vel_ref = x_ref_init[:, 21:24]
        joint_vel_ref = x_ref_init[:, 24:36]

        n = len(env_ids)
        full_pos = self._robot.data.default_joint_pos[env_ids].clone()
        full_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # Reference joint state + small random offset so the controller has something to correct
        joint_noise = torch.empty(n, len(self._joint_ids), device=self.device).uniform_(
            -self.cfg.init_noise_scale, self.cfg.init_noise_scale
        )
        full_pos[:, self._joint_ids] = full_pos[:, self._joint_ids] + joint_pos_rel + joint_noise
        full_vel[:, self._joint_ids] = joint_vel_ref

        # Root: reference xy/yaw (+ small noise, same role as joint_noise above)
        # + env origin; height stays at the default upright pose (z not tracked).
        xy_noise = torch.empty(n, 2, device=self.device).uniform_(
            -self.cfg.init_xy_noise_scale, self.cfg.init_xy_noise_scale
        )
        yaw_noise = torch.empty(n, device=self.device).uniform_(
            -self.cfg.init_yaw_noise_scale, self.cfg.init_yaw_noise_scale
        )
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :2] = self.scene.env_origins[env_ids, :2] + xy_ref + xy_noise
        zeros = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, yaw_ref + yaw_noise)
        root[:, 3:7] = quat

        # base twist is stored body-frame in x, but write_root_velocity_to_sim
        # expects world frame — rotate by the (yaw-only) orientation we just set.
        root_vel = torch.cat(
            [quat_apply(quat, base_lin_vel_ref), quat_apply(quat, base_ang_vel_ref)], dim=-1
        )

        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(full_pos, full_vel, None, env_ids)
