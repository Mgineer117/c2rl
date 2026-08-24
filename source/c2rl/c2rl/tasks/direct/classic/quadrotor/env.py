"""Quadrotor tracking environment."""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

STATE_NAMES = (
    "pos_x", "pos_y", "pos_z",
    "vel_x_w", "vel_y_w", "vel_z_w",
    "thrust", "roll", "pitch", "yaw",
)
# angle_idx / pos_dimension are derived from STATE_NAMES by BaseEnv (see
# agents/skrl/state_symmetry.py) -- the names are the single source of truth for
# which dims wrap, which translate, and which co-rotate under a yaw rotation.

G = 9.81

_X10_LIM = math.pi / 3
_X9_LIM = math.pi / 3
_X8_LIM = math.pi / 3
_X7_LOW = 0.5 * G
_X7_HIGH = 2 * G
_X4_LIM = 1.5
_X5_LIM = 1.5
_X6_LIM = 1.5

X_MIN = [-30.0, -30.0, -30.0, -_X4_LIM, -_X5_LIM, -_X6_LIM, _X7_LOW, -_X8_LIM, -_X9_LIM, -_X10_LIM]
X_MAX = [30.0, 30.0, 30.0, _X4_LIM, _X5_LIM, _X6_LIM, _X7_HIGH, _X8_LIM, _X9_LIM, _X10_LIM]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-5.0, -5.0, -5.0, -1.0, -1.0, -1.0, G, 0.0, 0.0, 0.0]
XREF_INIT_MAX = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, G, 0.0, 0.0, 0.0]

# Per-dimension, sized by how much of the +-6 actuator budget each dim's error
# consumes through K = (1/r)B^T M (measured along the CV-STEM tube: horizontal
# position/velocity and roll/pitch together account for ~80%, while pos_z /
# vel_z / thrust are cheap because thrust acts on them almost directly, and yaw
# is ~3%). u reaches control B only via u -> {thrust, roll, pitch} -> accel ->
# vel -> pos, so the relative-degree-3 horizontal chain is what saturates first;
# a uniform +-0.5 spent its budget there and left none for a usable lambda.
XE_INIT_MIN = [-0.15, -0.15, -0.3, -0.15, -0.2, -0.3, -0.3, -0.15, -0.15, -0.5]
XE_INIT_MAX = [0.15, 0.15, 0.3, 0.15, 0.2, 0.3, 0.3, 0.15, 0.15, 0.5]

_lim = 1.0
XE_MIN = [-_lim] * 10
XE_MAX = [_lim] * 10

# u = [d(thrust)/dt (m/s^3), roll/pitch/yaw rate (rad/s)] -- B puts ones on rows
# 6..9, so u drives the rates of thrust and attitude, not forces. The applied box
# is 2x this (env_base.py:37).
# thrust rate: thrust spans _X7_LOW..._X7_HIGH = 0.5g..2g = 4.9..19.6 m/s^2; at
# the old +-6 applied, traversing that range took 2.4 s, while a real quadrotor
# does it in ~0.1 s (~150 m/s^3). +-60 applied crosses it in 0.25 s.
# body rates: +-10 applied = 570 deg/s, between a gentle vehicle (~2-3 rad/s) and
# acro (~14 rad/s). The old +-6 was fine on its own, but paired with the crippled
# thrust rate the relative-degree-3 horizontal chain had no authority left.
UREF_MIN = [-30.0, -5.0, -5.0, -5.0]
UREF_MAX = [30.0, 5.0, 5.0, 5.0]

ENV_CONFIG = {
    "x_min": X_MIN, "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN, "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN, "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN, "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN, "uref_max": UREF_MAX,
    "num_dim_x": 10, "num_dim_control": 4,
    "dt": 0.025, "time_bound": 10.0,
    "q": 1.0, "r": 0.0,
}

class QuadrotorEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "quadrotor"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        force, theta_x, theta_y = x[:, 6], x[:, 7], x[:, 8]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = x[:, 3]
        f[:, 1] = x[:, 4]
        f[:, 2] = x[:, 5]
        f[:, 3] = -force * torch.sin(theta_y)
        f[:, 4] = force * torch.cos(theta_y) * torch.sin(theta_x)
        f[:, 5] = G - force * torch.cos(theta_y) * torch.cos(theta_x)
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 6, 0] = 1.0
        B[:, 7, 1] = 1.0
        B[:, 8, 2] = 1.0
        B[:, 9, 3] = 1.0
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            s_val = math.sin(freq * _t / self.time_bound * 2 * math.pi)
            uref[:, 0] += weight[:, 0] * s_val
            uref[:, 1] += weight[:, 1] * s_val
            uref[:, 2] += weight[:, 2] * s_val
            uref[:, 3] += weight[:, 3] * s_val
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 11))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        # 0.1, from scripts sizing the excitation against the box: at the old
# amplitude the reference clamped 47% of steps (vel_x_w/vel_y_w) and the
# stored (xref, uref) stopped being a trajectory of the plant.
        # excitation sized against the box: at 0.1 the reference clamped 47% of steps on
        # vel_x_w/vel_y_w, so (xref, uref) stopped being a trajectory.
        weights = 0.01 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
