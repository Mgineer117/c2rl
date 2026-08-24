from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ..common.path_tracking_base import PathTrackingBase
from .env_cfg import ManipulatorPathTrackingEnvCfg


class ManipulatorPathTrackingEnv(PathTrackingBase):
    """Franka Panda path-tracking environment.

    obs  = [x(21), x_ref(21), u_ref(7)] = 49
    reward = -||x - x_ref||^2

    x layout: joint_pos(7) + joint_vel(7) + ee_pos_local(3)
               + ee_lin_vel(3) + ee_yaw_vel(1)
    u layout: normalised joint position targets in [-1, 1] (7)
    """

    cfg: ManipulatorPathTrackingEnvCfg

    def __init__(self, cfg: ManipulatorPathTrackingEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._arm_ids, _ = self._robot.find_joints(self.cfg.arm_joint_names)
        self._ee_id, _ = self._robot.find_bodies(self.cfg.ee_body_name)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
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
        limits = self._robot.data.soft_joint_pos_limits[:, self._arm_ids, :]
        mid = 0.5 * (limits[..., 0] + limits[..., 1])
        half_range = 0.5 * (limits[..., 1] - limits[..., 0])
        self._joint_targets = mid + half_range * self._actions

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._arm_ids)

    @property
    def state_names(self) -> tuple[str, ...]:
        """Matches _get_physical_state's order. Deliberately uses ee_* rather
        than pos_*: the arm base is fixed, so the end-effector position is a
        function of joint_pos, not a free translation direction -- naming it
        pos_x/pos_y would wrongly invite the quotient to drop it. No base yaw
        either, so this env gets no symmetry reduction (mode="none").
        """
        n_a = len(self._arm_ids)
        return (
            tuple(f"joint_pos_{i}" for i in range(n_a))
            + tuple(f"joint_vel_{i}" for i in range(n_a))
            + ("ee_pos_x", "ee_pos_y", "ee_pos_z")
            + ("ee_lin_vel_x", "ee_lin_vel_y", "ee_lin_vel_z")
            + ("ee_yaw_vel",)
        )

    def _get_physical_state(self) -> torch.Tensor:
        ee_pos = self._robot.data.body_pos_w[:, self._ee_id[0], :] - self.scene.env_origins
        ee_lin_vel = self._robot.data.body_lin_vel_w[:, self._ee_id[0], :]
        ee_yaw_vel = self._robot.data.body_ang_vel_w[:, self._ee_id[0], 2:3]
        return torch.cat(
            [
                self._robot.data.joint_pos[:, self._arm_ids],
                self._robot.data.joint_vel[:, self._arm_ids],
                ee_pos,
                ee_lin_vel,
                ee_yaw_vel,
            ],
            dim=-1,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Path tracking never terminates early — non-finite/diverged states are
        # handled by the carry-forward guard in _sanitize_state instead (see
        # PathTrackingBase). Manipulators have no "fell" concept.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _set_robot_state_from_ref(self, env_ids: torch.Tensor, x_ref_init: torch.Tensor) -> None:
        # x: [joint_pos(7), joint_vel(7), ee_pos(3), ee_lin_vel(3), ee_yaw_vel(1)]
        joint_pos = x_ref_init[:, 0:7]
        joint_vel = x_ref_init[:, 7:14]

        root = self._robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
