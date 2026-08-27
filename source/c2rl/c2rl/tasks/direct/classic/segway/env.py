"""Segway tracking environment (ported to batched PyTorch)."""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# Denote angle indices to handle smooth transition
STATE_NAMES = ("pos_x", "pitch", "vel_x_b", "pitch_rate")
# angle_idx / pos_dimension are derived from STATE_NAMES by BaseEnv (see
# agents/skrl/state_symmetry.py) -- the names are the single source of truth for
# which dims wrap, which translate, and which co-rotate under a yaw rotation.

# X bounds
#
# PITCH_LIM is 0.90 rad (51.6 deg), not the pi/3 (60 deg) it was, and that is a
# statement about what is CERTIFIABLE, not about the plant.
#
# lambda*(x) over the old box is a smooth 0.19-0.34 across ~93% of it and
# collapses in exactly two opposite corners: maximum tilt combined with fast
# rotation THE OTHER WAY (pitch -1.047 with pitch_rate +2.69 gives 0.00367, and
# the mirror +1.047 / -2.69 gives 0.00939). A uniform rate must hold everywhere,
# so those two corners set it for the whole box -- a 93x collapse driven by ~1%
# of the volume. At lambda = 0.00367 over a 15 s episode the certified bound is
# C*exp(-lambda*T) = 1.15 * 0.946 = 1.088 > 1: it does not even promise the error
# ends below where it started. The certificate was vacuous.
#
# Trimming pitch by one grid column removes both corners and lifts the cap 11x,
# to 0.0407, where the bound is 0.625 -- a real 37% guaranteed decay -- while
# keeping 87% of the (pitch, pitch_rate) area. Trimming pitch_rate instead is a
# worse trade: 6x for 27% of the area. See scripts/minproj_plot.py's cached grid
# and the table in the 2026-08-22 session.
#
# This does NOT touch the initial distribution: XE_INIT draws pitch from +-0.15,
# far inside either bound. What it changes is the set the CM dataset certifies
# over, and the tilt at which an episode terminates (60 deg -> 51.6 deg).
PITCH_LIM = 0.90
X_MIN = [-5.0, -PITCH_LIM, -1.0, -math.pi]
X_MAX = [5.0, PITCH_LIM, 1.0, math.pi]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

# Initial reference state bounds
XREF_INIT_MIN = [0.0, 0, 0.0, 0]
XREF_INIT_MAX = [0.0, 0, 0.0, 0]

# Initial perturbation to the reference state.
# pitch and pitch_rate used to span the entire state box (+-pi/3 and +-pi, i.e.
# X_MIN/X_MAX exactly), so reset could place the segway at a 60-degree tilt
# spinning at pi rad/s -- mid-fall, not a tracking error. Recovering 60 degrees
# alone needs |u|=4.17 of the +-6 available, and along the CV-STEM tube
# pitch_rate accounted for ~49% of the feedback demand and pitch ~19%, with the
# peak demand 5.3x over budget. Tightened to a genuine perturbation.
XE_INIT_MIN = [-1.0, -0.15, -0.15, -0.5]
XE_INIT_MAX = [1.0, 0.15, 0.15, 0.5]

# initial reference state perturbation bounds for c3m
lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds. The applied box is 2x this (env_base.py:37). u is a
# normalized wheel torque with no direct hardware unit, so anchor it through its
# effect: B[2,0] = (-1.8 cos(th) - 10.9)/(cos(th) - 24.7) ~ 0.54 maps u to the
# body-frame acceleration v_dot, so +-12 applied is +-6.4 m/s^2 of wheel accel --
# in range for a two-wheel balancer (peak ~5-10 m/s^2), where the old +-6 applied
# capped it at 3.2 m/s^2. For scale, statically holding the XE_INIT pitch of
# 0.15 rad needs only |u| = 0.65 and a 60-degree tilt needs 4.17; the box has to
# cover the pitch_rate term too, which is what actually saturated it.
UREF_MIN = [-6.0]
UREF_MAX = [6.0]

# Initial state drawn directly from this box (see env_base.X_INIT_MIN),
# placing every episode start in the plant's low-lambda region:
# lbd*(x0) median 0.2207 -> 0.0987. The sign dims are
# mirrored by one shared sign, because the slow set is two lobes on a
# diagonal that no single box can cover.
# The pitch ceiling is PITCH_LIM, not a literal: it used to be 1.0467, the old
# pi/3 box bound, and when the box moved to 0.90 the 37.9% of this range above
# 0.90 was clamped straight onto the box face. That is not a cosmetic artifact --
# x_0 pinned to a boundary is not the draw this box describes, and every
# Stability/* metric is normalised by the error at x_0. Derive it so the two
# cannot drift apart again (scripts/check_boxes.py fails loudly if they do).
X_INIT_MIN = [-5.0, 0.6596, 0.0098, -3.1244]
X_INIT_MAX = [5.0, PITCH_LIM, 0.994, -0.6786]
X_INIT_SIGN_DIMS = [1, 2, 3]

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
    # 60 s, not 15. At the certified lambda = 0.0514 an exponential decay needs
    # ln(20)/lambda = 58.3 s to bring the error to 5% of its initial value; 15 s
    # reached only exp(-0.771) = 46%, so the episode ended while the error was
    # still half its starting size. dt stays 0.03: at 0.12 (the value that would
    # keep 500 steps) forward-Euler error over 2 s open-loop goes from 15% to 41%
    # on this unstable plant, and lambda*dt from 0.096 to 0.383.
    #
    # The CM dataset does NOT need regenerating: its cache key is
    # (lbd, w_lb, w_ub, eps, solver, N, r_scaler, chi/nu_weight, wdot_dt,
    # random_ratio, wdot_trajectory, temporal_dt) -- no env dt and no time_bound --
    # and the SDP samples the state box with continuous-time Jacobians, so lambda
    # is a rate in 1/s that episode length never enters. Verified by preflight.
    "time_bound": 60.0,
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

class SegwayEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "segway"
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
            torch.cos(theta) * (9.8 * torch.sin(theta) + 11.5 * v)
            + 68.4 * v
            - 1.2 * (omega**2) * torch.sin(theta)
        ) / (torch.cos(theta) - 24.7)
        f[:, 3] = (
            -58.8 * v * torch.cos(theta)
            - 243.5 * v
            - torch.sin(theta) * (208.3 + (omega**2) * torch.cos(theta))
        ) / (torch.cos(theta) ** 2 - 24.7)
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 0] = (-1.8 * torch.cos(theta) - 10.9) / (torch.cos(theta) - 24.7)
        B[:, 3, 0] = (9.3 * torch.cos(theta) + 38.6) / (torch.cos(theta) ** 2 - 24.7)
        return B

    def _B_null_logic(self, x):
        n = x.shape[0]
        theta = x[:, 1]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = (9.3 * torch.cos(theta) + 38.6) / (torch.cos(theta) ** 2 - 24.7)
        Bbot[:, 3, 2] = -(-1.8 * torch.cos(theta) - 10.9) / (torch.cos(theta) - 24.7)
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
