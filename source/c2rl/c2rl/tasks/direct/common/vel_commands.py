"""Velocity command generator: constant (vx, vy, vz) + sinusoidal yaw rate.

Each env gets independent commands sampled at reset.
Sinusoidal yaw makes the robot carve curved trajectories rather than
straight lines, giving richer coverage of the state space.

    yaw_rate(t) = A * sin(omega * t + phi)

where A, omega, phi are sampled uniformly at reset.

Markov note: ``get()`` also returns (A, omega, sin(phase), cos(phase)) — the
scalar yaw_rate value alone does not determine its own future (the same value
is consistent with many (A, omega, phase) combinations, and even knowing all
three, a single sin() value has two solutions per period with opposite
derivative sign). Exposing the full generator (continuously embedded via
sin/cos, matching this repo's angle-embedding convention, rather than a raw
ever-growing `phase`) makes the observation Markov w.r.t. the command's own
future — needed for GAE/Q-learning value estimates to be well-defined, even
though the reward itself only ever needs the instantaneous value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class VelCmdCfg:
    # linear velocity ranges [m/s], constant per episode
    vx_range: tuple[float, float] = (0.5, 1.5)    # always forward → trajectories go "away"
    vy_range: tuple[float, float] = (-0.2, 0.2)   # small lateral drift
    vz_range: tuple[float, float] = (0.0, 0.0)    # zero for flat-ground locomotion

    # sinusoidal yaw parameters — slower frequency keeps trajectories from looping back
    yaw_A_range: tuple[float, float] = (0.2, 0.8)    # amplitude  [rad/s]
    yaw_omega_range: tuple[float, float] = (0.3, 1.0) # frequency  [rad/s]
    # If True, yaw_omega is sampled as a binary {lo, hi} choice (50/50) instead
    # of continuous uniform — e.g. lo=0.0 (constant yaw rate, no oscillation)
    # vs hi=one-cycle-per-episode, with nothing in between.
    yaw_omega_binary: bool = False
    # phase phi [rad] sampled uniformly from this range. Default (0, 2π) keeps the
    # original random-start behavior; set to (0.0, 0.0) to start every episode at
    # zero yaw rate (a deterministic S-weave that returns to the initial heading
    # after exactly one full cycle).
    yaw_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)


class VelCommands:
    """
    Manages per-environment velocity commands.

    Usage in env:
        self._cmd = VelCommands(self.num_envs, self.device, self.cfg.vel_cmd)

    In _reset_idx:
        self._cmd.reset(env_ids)

    In _get_observations (or _get_rewards):
        t = self.episode_length_buf.float() * self.step_dt   # seconds
        cmds = self._cmd.get(t)   # (N, 8): [vx, vy, vz, yaw_rate, A, omega, sin(phase), cos(phase)]
    """

    def __init__(self, num_envs: int, device: str, cfg: VelCmdCfg):
        self.cfg = cfg
        self.device = device

        self.vx = torch.zeros(num_envs, device=device)
        self.vy = torch.zeros(num_envs, device=device)
        self.vz = torch.zeros(num_envs, device=device)

        self.yaw_A = torch.zeros(num_envs, device=device)
        self.yaw_omega = torch.zeros(num_envs, device=device)
        self.yaw_phi = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        def _u(lo: float, hi: float) -> torch.Tensor:
            return torch.empty(n, device=self.device).uniform_(lo, hi)

        lo, hi = self.cfg.vx_range
        self.vx[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.vy_range
        self.vy[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.vz_range
        self.vz[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.yaw_A_range
        self.yaw_A[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.yaw_omega_range
        if self.cfg.yaw_omega_binary:
            choice = (torch.rand(n, device=self.device) < 0.5).float()  # 0 -> lo, 1 -> hi
            self.yaw_omega[env_ids] = lo + choice * (hi - lo)
        else:
            self.yaw_omega[env_ids] = _u(lo, hi)
        lo, hi = self.cfg.yaw_phase_range
        self.yaw_phi[env_ids] = _u(lo, hi)

    def get(self, episode_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            episode_time: (N,) seconds elapsed since last reset per env.
        Returns:
            (N, 8) tensor: [vx, vy, vz, yaw_rate, A, omega, sin(phase), cos(phase)]
            The last 4 fully determine the yaw-rate generator's future (A,
            omega fixed per episode; sin/cos of the current phase resolve the
            "which side of the sine curve" ambiguity a raw yaw_rate value
            can't) — see module docstring's Markov note.
        """
        phase = self.yaw_omega * episode_time + self.yaw_phi
        yaw_rate = self.yaw_A * torch.sin(phase)
        return torch.stack(
            [self.vx, self.vy, self.vz, yaw_rate,
             self.yaw_A, self.yaw_omega, torch.sin(phase), torch.cos(phase)],
            dim=-1,
        )
