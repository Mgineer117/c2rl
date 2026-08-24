"""AUV surge/pitch tracking environment (batched PyTorch).

Reference
---------
T. I. Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control",
Wiley, 2011, Ch. 7-8 (manoeuvring model; quadratic damping and fin lift
proportional to the square of the relative speed).

Why this plant is here
----------------------
category a, dynamic-pressure scaling -- the most extreme case in this repo.
Stern-plane authority is proportional to 0.5*rho*V^2, and an AUV's operating
speed range is much wider relatively than an aircraft's: at 0.3 m/s the fins do
essentially nothing, at 3.0 m/s they dominate.  Over a 10x speed range the fin
moment varies 100x, and the measured sv(B) spread is 98.691 -- versus the
aircraft's 17.249, whose envelope is only 4.2x in speed.

The thruster (surge) keeps constant authority, so this plant also has the
property that the two input channels scale completely differently with state.
The certificate over the whole envelope is therefore governed by the slow,
almost-unactuated-in-pitch corner, which is exactly the regime a state subset
can exclude.
"""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# AUV parameters (Fossen 2011, small survey-class vehicle)
RHO = 1025.0        # kg/m^3, seawater
MASS = 80.0         # kg (including added mass in surge)
IYY = 40.0          # kg m^2 (including added inertia in pitch)
A_FRONT = 0.05      # m^2, frontal area
C_D = 0.30          # drag coefficient
S_FIN = 0.05        # m^2, stern-plane area
L_ARM = 0.50        # m, fin moment arm
C_M_DELTA = 0.80    # 1/rad, fin lift-curve slope
C_Q = 5.0           # N m s/rad, pitch damping
BG_RESTORE = 12.0   # N m/rad, metacentric restoring moment (righting couple)

# Inputs are normalised commands in [-1, 1]; u[0] = 1 is full stern plane
# (DELTA_MAX rad), u[1] = 1 is full thrust (THRUST_MAX N). Same reason as the
# aircraft: R = r*I cannot weight a fin deflection and a thruster force sensibly
# when their raw units differ by two orders of magnitude, and the measured
# demand was 56 RAD of stern plane. Surge speed is already O(1) so the state
# needs no scaling -- only the inputs do.
DELTA_MAX = 0.30     # rad
THRUST_MAX = 25.0    # N

STATE_NAMES = ("pitch", "vel", "pitch_rate")

# X bounds. The 10x speed range is the whole point -- narrowing it would remove
# the dynamic-pressure variation this plant exists to demonstrate.
X_MIN = [-0.40, 0.30, -0.50]
X_MAX = [0.40, 3.00, 0.50]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-0.10, 1.00, -0.10]
XREF_INIT_MAX = [0.10, 2.00, 0.10]

XE_INIT_MIN = [-0.10, -0.30, -0.10]
XE_INIT_MAX = [0.10, 0.30, 0.10]

lim = 1.0
XE_MIN = [-lim, -lim, -lim]
XE_MAX = [lim, lim, lim]

# reference control bounds -- [stern plane (rad), thruster (N)]. Applied box 2x.
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
    "num_dim_x": 3,
    "num_dim_control": 2,
    "dt": 0.02,
    "time_bound": 10.0,
    "q": 1.0,
    "r": 0.0,
}


class AUVEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "auv"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        n = x.shape[0]
        theta, V, q = x[:, 0], x[:, 1], x[:, 2]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = q
        # quadratic drag -- the dominant surge dynamic for a streamlined hull
        f[:, 1] = -0.5 * RHO * C_D * A_FRONT * V * torch.abs(V) / MASS
        # pitch damping plus the hydrostatic righting moment
        f[:, 2] = (-C_Q * q - BG_RESTORE * torch.sin(theta)) / IYY
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        V = x[:, 1]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        # thruster: Constant authority
        B[:, 1, 1] = THRUST_MAX / MASS
        # stern plane: authority ~ V^2, a 100x swing over the speed range
        B[:, 2, 0] = 0.5 * RHO * V ** 2 * S_FIN * L_ARM * C_M_DELTA * DELTA_MAX / IYY
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
        weights = 0.08 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
