"""PVTOL (planar vertical take-off and landing) tracking environment.

Reference
---------
J. Hauser, S. Sastry and G. Meyer, "Nonlinear control design for slightly
non-minimum phase systems: application to V/stol aircraft", Automatica,
28(4):665-679, 1992.  The epsilon-coupled model below (rolling moment produces a
small parasitic lateral force) is that paper's, and epsilon is its coupling
coefficient.

Why this plant is here
----------------------
category c, the negative control.  B(x) is state-dependent -- it carries
-sin(phi) and cos(phi) -- and yet

    || (-sin(phi), cos(phi)) || = 1   for every phi,

so the thrust column rotates with attitude and never shrinks.  Measured sv(B)
spread 1.000, identical to the kinematic car and the velocity-controlled
turtlebot.  No state is harder to actuate than any other, so no state subset can
certify a faster contraction rate and subset pruning is useless here.

This is the plant that separates the real criterion from the obvious one: the
question is never "does B depend on x" but "do B's singular values depend on x".
Keeping a plant that answers no is what makes the positive results (cartpole
2.213, ball-and-beam 3.470, two-link arm 5.753) mean something.
"""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# PVTOL parameters (Hauser, Sastry & Meyer 1992)
G = 9.81
EPS = 0.1        # thrust/rolling-moment coupling; the paper's "slightly
                 # non-minimum phase" parameter

STATE_NAMES = ("pos_x", "pos_y", "roll", "vel_x_w", "vel_y_w", "roll_rate")

X_MIN = [-4.0, -4.0, -0.6, -3.0, -3.0, -3.0]
X_MAX = [4.0, 4.0, 0.6, 3.0, 3.0, 3.0]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-1.0, -1.0, -0.1, -0.3, -0.3, -0.3]
XREF_INIT_MAX = [1.0, 1.0, 0.1, 0.3, 0.3, 0.3]

XE_INIT_MIN = [-0.5, -0.5, -0.15, -0.3, -0.3, -0.3]
XE_INIT_MAX = [0.5, 0.5, 0.15, 0.3, 0.3, 0.3]

lim = 1.0
XE_MIN = [-lim] * 6
XE_MAX = [lim] * 6

# reference control bounds -- [thrust, rolling moment]. The applied box is 2x
# this. Hovering needs thrust = g = 9.81, so the reference box is centred on
# hover rather than on zero and reaches 2g at full deflection.
UREF_MIN = [0.0, -2.0]
UREF_MAX = [2.0 * G, 2.0]

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
    "num_dim_x": 6,
    "num_dim_control": 2,
    "dt": 0.02,
    "time_bound": 10.0,
    "q": 1.0,
    "r": 0.0,
}


class PVTOLEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "pvtol"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        n = x.shape[0]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = x[:, 3]          # xdot
        f[:, 1] = x[:, 4]          # ydot
        f[:, 2] = x[:, 5]          # phidot
        f[:, 4] = -G               # gravity is the only drift on the velocities
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        phi = x[:, 2]
        s, c = torch.sin(phi), torch.cos(phi)
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        B[:, 3, 0] = -s            # thrust tilts with attitude ...
        B[:, 4, 0] = c             # ... but ||(-s, c)|| == 1 always
        B[:, 3, 1] = EPS * c       # parasitic lateral force from the moment
        B[:, 4, 1] = EPS * s
        B[:, 5, 1] = 1.0
        return B

    def _B_null_logic(self, x):
        """Null space of B^T, computed in closed form.

        The default implementation puts the identity on the leading
        ``x_dim - u_dim`` rows, which assumes B is confined to the trailing
        rows. Here B occupies rows 3-5, so that default would not be orthogonal
        to it. Three null directions are the unactuated positions/attitude
        (e0, e1, e2); the fourth lies inside span(e3, e4, e5) and is
        (0, 0, 0, cos(phi), sin(phi), -eps) -- orthogonal to both columns:
            (-s)(c) + (c)(s)            = 0
            (eps c)(c) + (eps s)(s) - eps = 0
        """
        n = x.shape[0]
        phi = x[:, 2]
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = 1.0
        Bbot[:, 1, 1] = 1.0
        Bbot[:, 2, 2] = 1.0
        Bbot[:, 3, 3] = torch.cos(phi)
        Bbot[:, 4, 3] = torch.sin(phi)
        Bbot[:, 5, 3] = -EPS
        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        # Trim: thrust must cancel gravity along the body axis, so it grows as
        # 1/cos(phi) when tilted. A constant G only hovers at phi = 0; tilted, it
        # under-lifts and the reference falls out of the box (measured 36-53%
        # clamping at every excitation amplitude).
        xr = infos.get("xref_t")
        phi = xr[:, 2] if xr is not None else torch.zeros(n, device=self.device)
        uref[:, 0] = G / torch.cos(phi).clamp(min=0.3)
        if xr is not None:
            # Position and attitude are pure double integrators here (A is
            # nilpotent, A^2 = 0), so hover thrust alone leaves the reference
            # drifting until it clamps -- 36-53% of steps at every amplitude.
            # Stabilise about the start pose: altitude on the thrust channel,
            # lateral position through attitude on the moment channel.
            p0 = infos["xref_0"]
            uref[:, 0] += -4.0 * (xr[:, 1] - p0[:, 1]) - 3.0 * xr[:, 4]
            phi_cmd = 0.25 * (xr[:, 0] - p0[:, 0]) + 0.35 * xr[:, 3]
            uref[:, 1] += -6.0 * (phi - phi_cmd.clamp(-0.4, 0.4)) - 3.0 * xr[:, 5]
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
        weights = 0.5 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
