"""Cartpole tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# Cartpole parameters
mc = 1.0
mp = 1.0
g = 9.81
l = 1.0

# Denote angle indices to handle smooth transition
STATE_NAMES = ("pos_x", "pitch", "vel_x_b", "pitch_rate")
# angle_idx / pos_dimension are derived from STATE_NAMES by BaseEnv (see
# agents/skrl/state_symmetry.py) -- the names are the single source of truth for
# which dims wrap, which translate, and which co-rotate under a yaw rotation.

# X bounds
X_MIN = [-5.0, -math.pi / 3, -1.0, -1]
X_MAX = [5.0, math.pi / 3, 1.0, 1]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# Initial reference state bounds
XREF_INIT_MIN = [0.0, 0, 0.0, 0]
XREF_INIT_MAX = [0.0, 0, 0.0, 0]

# Initial perturbation to the reference state
lim = 0.3
XE_INIT_MIN = [-lim, -lim, -lim, -lim]
XE_INIT_MAX = [lim, lim, lim, lim]

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds. u is a force on the cart (N); the applied box is 2x
# this (env_base.py:37). Statically holding the pole at tilt theta needs
# u = (mc+mp)*g*tan(theta) = 19.62*tan(theta), so the old +-6 applied could not
# hold even the XE_INIT pitch of 0.3 rad (19.62*tan(0.3) = 6.07 N) -- reset could
# place the pole past what any gain could recover, making the actuator check
# infeasible by arithmetic, not by the SDP. +-12 N applied holds 0.55 rad and
# matches real hardware (Gym cartpole uses 10 N, Quanser's linear pendulum ~15).
UREF_MIN = [-6.0]
UREF_MAX = [6.0]

# Initial state drawn directly from this box (see env_base.X_INIT_MIN),
# placing every episode start in the plant's low-lambda region:
# lbd*(x0) median 1.3194 -> 0.5625. The sign dims are
# mirrored by one shared sign, because the slow set is two lobes on a
# diagonal that no single box can cover.
X_INIT_MIN = [-5.0, 0.8286, -1.0, 0.0168]
X_INIT_MAX = [5.0, 1.0467, 1.0, 0.9676]
X_INIT_SIGN_DIMS = [1, 3]

ENV_CONFIG = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "x_init_min": X_INIT_MIN,
    "x_init_max": X_INIT_MAX,
    "x_init_sign_dims": X_INIT_SIGN_DIMS,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 4,
    "num_dim_control": 1,
    "dt": 0.03,
    "time_bound": 15.0,
    # Episodes run the full horizon rather than ending on the first excursion
    # from the termination box. This plant is unstable and pi starts from random
    # init (u = uref + pi, no warm-start), so with the box armed every episode
    # ended after ~5 of 500 steps, and stability_summary() then withholds AUC /
    # contraction rate / overshoot / score entirely -- publishing only
    # early_end_frac -- because no episode survives to be measured. Running the
    # full horizon lets AUC measure the divergence instead of the metric being
    # unavailable.
    #
    # Safe on the reward side: the excursion was reported as TRUNCATION, not
    # termination, and every c2rl-ppo yaml sets time_limit_bootstrap: true, so
    # there was never a suicide bonus to lose here (see termination_box.py).
    # The cost is the one that box existed to prevent: a diverged episode now
    # contributes its remaining off-distribution steps to the rollout batch,
    # which is a known source of seed-to-seed variance on these two envs.
    "terminate_out_of_box": False,
    "q": 1.0,
    "r": 0.0,
}

class CartPoleEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "cartpole"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device
        )

    def _f_logic(self, x):
        n = x.shape[0]
        p, theta, v, omega = [x[:, i] for i in range(self.num_dim_x)]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = v
        f[:, 1] = omega
        f[:, 2] = (
            mp * torch.sin(theta) * (l * (omega**2) - g * torch.cos(theta))
            / (mc + mp * (torch.sin(theta) ** 2))
        )
        f[:, 3] = (
            (mp * l * (omega**2) * torch.cos(theta) * torch.sin(theta) - (mc + mp) * g * torch.sin(theta))
            / l / (mc + mp * (torch.sin(theta) ** 2))
        )
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 0] = 1 / (mc + mp * (torch.sin(theta) ** 2))
        B[:, 3, 0] = torch.cos(theta) / l / (mc + mp * (torch.sin(theta) ** 2))
        return B

    def _B_null_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = torch.cos(theta) / l / (mc + mp * (torch.sin(theta) ** 2))
        Bbot[:, 3, 2] = -1.0 / (mc + mp * (torch.sin(theta) ** 2))
        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        xref_0 = infos["xref_0"]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        uref[:, 0] = 10.2 * xref_0[:, 2] / 47.9
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            term = weight[:, 0] * ((-1) ** int(freq * _t / self.time_bound)) * math.sin(freq * _t / self.time_bound * 2 * math.pi)
            uref[:, 0] += term
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = []
        n = len(env_ids)
        if len(freqs) > 0:
            weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
            weights = 0.0 * weights / torch.sqrt((weights**2).sum(dim=1, keepdim=True))
        else:
            weights = torch.zeros(n, 0, len(UREF_MIN), device=self.device)
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
