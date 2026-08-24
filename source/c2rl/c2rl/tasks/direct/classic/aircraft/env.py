"""Longitudinal aircraft tracking environment (batched PyTorch).

References
----------
B. L. Stevens and F. L. Lewis, "Aircraft Control and Simulation", 2nd ed.,
Wiley, 2003, Ch. 2-3 (longitudinal short-period / phugoid model in
[V, alpha, q, theta]).

W. J. Rugh and J. S. Shamma, "Research on gain scheduling", Automatica,
36(10):1401-1425, 2000 -- the survey explaining why a single linear controller
cannot cover a flight envelope, which is the same fact this plant contributes
here.

Why this plant is here
----------------------
category a, dynamic-pressure scaling, and the one with the strongest physical
story.  Control-surface authority is proportional to dynamic pressure

    qbar = 0.5 * rho * V^2

so the elevator's moment scales with the square of airspeed: at 60 m/s it has
(60/250)^2 = 1/17 of the authority it has at 250 m/s.  Measured sv(B) spread
17.249 over the state box below.

This is precisely why gain scheduling was invented in aerospace (Rugh & Shamma
above): a fixed gain that is stable at low speed is over-aggressive at high
speed, and vice versa.  A contraction certificate over the whole envelope must
be governed by the slow, low-authority corner -- which is exactly the situation
where restricting to a state subset buys a faster certified rate.
"""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# Aircraft parameters -- a generic medium transport (Stevens & Lewis Ch. 3)
RHO = 1.225          # kg/m^3, sea-level density (held constant: no altitude state)
S_WING = 25.0        # m^2
CBAR = 2.0           # m, mean aerodynamic chord
IYY = 2.5e4          # kg m^2
MASS = 5000.0        # kg
G = 9.81
C_L_ALPHA = 5.0      # 1/rad
C_D0 = 0.03
C_M0 = 0.0
C_M_ALPHA = -0.5     # 1/rad, statically stable
C_M_Q = -10.0        # pitch damping
C_M_DELTA = -1.2     # 1/rad, elevator effectiveness

# ── Non-dimensionalisation ──────────────────────────────────────────────── #
# Airspeed is carried as v = V / V_REF and the two inputs are normalised
# commands in [-1, 1] that map onto the physical ranges below.
#
# Without this the SDP is unusable, and not for a tunable reason: raw V spans
# 60-250 m/s while the angles span ~0.1 rad, and R = r*I applies one weight to
# both. The certified gain K = R^-1 B^T M then reads a 20 m/s speed error and a
# 0.1 rad attitude error through the same scale, so the elevator channel is
# dominated by speed and saturates at every r -- measured |u| p95 of 47571 RAD
# of elevator, versus |K| of 2-27 on the repo's other plants. Raising uref to
# cover that would certify a controller commanding physically impossible
# surfaces. Scaling the state and the inputs fixes the conditioning while
# leaving the physics, and the qbar ~ V^2 variation this plant exists to show,
# completely intact (sv(B) spread is unchanged).
V_REF = 100.0            # m/s, so v = V/V_REF is O(1)
DELTA_MAX = 0.20         # rad of elevator at u[0] = 1
THRUST_MAX = 10000.0     # N of thrust at u[1] = 1

# alpha is not in the symmetry vocabulary, which is correct: it is an
# aerodynamic angle in a small range, not a wrapping heading. State order puts
# the two actuated derivatives (V, q) last so the default B_null applies.
STATE_NAMES = ("alpha", "pitch", "vel", "pitch_rate")   # vel = V / V_REF

# X bounds. V spans a realistic envelope (60 = near stall, 250 = cruise-high),
# and that 4.2x speed range is what makes qbar vary 17x.
X_MIN = [-0.20, -0.40, 0.60, -0.50]
X_MAX = [0.30, 0.40, 2.50, 0.50]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-0.05, -0.10, 1.20, -0.10]
XREF_INIT_MAX = [0.05, 0.10, 1.80, 0.10]

# +-0.05 = +-5 m/s of speed error. The old +-20 m/s was not a tracking error,
# it was a different flight condition.
XE_INIT_MIN = [-0.05, -0.10, -0.05, -0.10]
XE_INIT_MAX = [0.05, 0.10, 0.05, 0.10]

lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds -- Normalised commands, u[0] = 1 is full elevator
# (DELTA_MAX rad) and u[1] = 1 is full thrust (THRUST_MAX N). Applied box is 2x,
# i.e. the physical surfaces can reach 2x their nominal range under feedback.
UREF_MIN = [-1.0, -1.0]
UREF_MAX = [1.0, 1.0]


ENV_CONFIG = {
    "x_min": X_MIN,
    "x_max": X_MAX,
    "x_termination_min": X_TERMINATION_MIN,
    "x_termination_max": X_TERMINATION_MAX,
    "xref_init_min": XREF_INIT_MIN,
    "xref_init_max": XREF_INIT_MAX,
    "xe_init_min": XE_INIT_MIN,
    "xe_init_max": XE_INIT_MAX,
    "xe_min": XE_MIN,
    "xe_max": XE_MAX,
    "state_names": STATE_NAMES,
    "uref_min": UREF_MIN,
    "uref_max": UREF_MAX,
    "num_dim_x": 4,
    "num_dim_control": 2,
    "dt": 0.02,
    "time_bound": 10.0,
    "q": 1.0,
    "r": 0.0,
}


class AircraftEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "aircraft"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        n = x.shape[0]
        alpha, theta, v, q = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        V = v * V_REF
        qbar = 0.5 * RHO * V ** 2
        lift = qbar * S_WING * C_L_ALPHA * alpha
        drag = qbar * S_WING * (C_D0 + 0.05 * alpha ** 2)
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = q - lift / (MASS * V) + (G / V) * torch.cos(theta - alpha)
        f[:, 1] = q
        f[:, 2] = (-drag / MASS - G * torch.sin(theta - alpha)) / V_REF
        f[:, 3] = qbar * S_WING * CBAR * (
            C_M0 + C_M_ALPHA * alpha + C_M_Q * q * CBAR / (2.0 * V)) / IYY
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        V = x[:, 2] * V_REF
        qbar = 0.5 * RHO * V ** 2
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 2, 1] = THRUST_MAX / (MASS * V_REF)               # thrust -> vdot
        # The point of this plant: elevator authority scales with V^2.
        B[:, 3, 0] = qbar * S_WING * CBAR * C_M_DELTA * DELTA_MAX / IYY
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            phase = math.sin(freq * _t / self.time_bound * 2 * math.pi)
            uref[:, 0] += weight[:, 0] * phase
            uref[:, 1] += weight[:, 1] * phase
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 6))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        weights = 0.05 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
