"""TORA (translational oscillator with rotational actuator) tracking environment.

Reference
---------
Z. P. Jankovic, D. Fontaine and P. V. Kokotovic, "TORA example: cascade- and
passivity-based control designs", IEEE Transactions on Control Systems
Technology, 4(3):292-297, 1996.  Also R. T. Bupp, D. S. Bernstein and V. T.
Coppola, "A benchmark problem for nonlinear control design", International
Journal of Robust and Nonlinear Control, 8:307-310, 1998, which fixed the
normalised form used below.  A cart on a spring carrying an actuated
eccentric rotor; the only actuation is the rotor torque, and the cart is moved
purely through the rotational-translational coupling.

Why this plant is here
----------------------
the negative control that actually certifies.  B(x) is genuinely state
dependent -- the coupling ``eps*cos(theta)`` sits in the inertia matrix, so
``B = M(theta)^-1 [0;1]`` moves with the rotor angle -- and yet its singular
values barely move: measured sv(B) spread 1.062, against cartpole's 2.213 and
the two-link arm's 5.035.

That combination is what the repo was missing.  pvtol and the velocity-
controlled turtlebot also have a state-dependent B with sv spread 1.000, but
neither can be CV-STEM-certified at all (pvtol's linearisation is unstabilisable,
ctrb rank 4/6; turtlebot is driftless), so neither can demonstrate "B depends on
x, yet a state subset buys nothing" -- that claim needs a feasible baseline
lambda to compare against.  TORA has one: its spring term couples configuration
back into acceleration, so ctrb rank is 4/4 at every sampled state.

Reading it: B enters the CV-STEM LMI only through ``nu*(2/r)*B B^T`` with ``nu``
shared across states, so excluding a state pays only if that state is genuinely
weaker to actuate.  A B that changes direction, or whose magnitude barely moves,
offers nothing to exclude.  Note the predictor is directional, not proportional:
segway's spread of 1.139 yielded a 5.63x subset gain while cartpole's 2.208
yielded 2.52x, so sv(B) spread says whether a gain exists, not how large.
"""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# TORA parameters -- Bupp et al.'s normalised benchmark. In these coordinates
# the cart mass, spring constant and rotor inertia are all scaled to 1, and the
# single remaining parameter is the eccentricity coupling.
EPS = 0.2      # rotational/translational coupling; |EPS| < 1 keeps M(theta) > 0

STATE_NAMES = ("joint_pos_0", "joint_vel_0", "pitch", "pitch_rate")
# joint_pos_0 rather than pos_x: the spring makes the dynamics depend on the
# cart's absolute displacement, so this plant is not translation invariant.
# "pitch" carries the rotor angle so it is angle-wrapped.

X_MIN = [-1.0, -2.0, -math.pi, -2.0]
X_MAX = [1.0, 2.0, math.pi, 2.0]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-0.3, -0.3, -0.5, -0.3]
XREF_INIT_MAX = [0.3, 0.3, 0.5, 0.3]

XE_INIT_MIN = [-0.2, -0.2, -0.3, -0.2]
XE_INIT_MAX = [0.2, 0.2, 0.3, 0.2]

lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds -- normalised rotor torque. Applied box is 2x this.
# +-10 rather than the +-1 that swings the rotor alone: the rotor is the only
# actuator, and it reaches the cart through the weak coupling eps = 0.2, so
# damping the (undamped) spring costs roughly 1/eps times the torque the rotor
# needs for itself. Measured: at +-1 the certified gain |K| = 38 puts 65% of
# controls outside the box, at +-10 it is 0.7%. These are Bupp et al.'s
# Normalised units -- mass, spring and inertia all scaled to 1 -- so there is no
# hardware torque limit being violated here.
UREF_MIN = [-10.0]
UREF_MAX = [10.0]

# Initial state drawn directly from this box (see env_base.X_INIT_MIN),
# placing every episode start in the plant's low-lambda region:
# lbd*(x0) median 0.7334 -> 0.4160. The sign dims are
# mirrored by one shared sign, because the slow set is two lobes on a
# diagonal that no single box can cover.
X_INIT_MIN = [0.0337, -2.0, -2.1151, 0.051]
X_INIT_MAX = [0.9566, 2.0, -1.1153, 0.2354]
X_INIT_SIGN_DIMS = [0, 2, 3]

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
    "dt": 0.02,
    "time_bound": 10.0,
    "q": 1.0,
    "r": 0.0,
}


def _inv_inertia(theta):
    """M(theta)^-1 for M = [[1, eps cos], [eps cos, 1]], in closed form.

    det = 1 - eps^2 cos^2(theta) is bounded away from 0 for |eps| < 1, so the
    inverse never blows up -- which is exactly why sigma(B) moves so little.
    """
    c = torch.cos(theta)
    det = 1.0 - (EPS * c) ** 2
    return 1.0 / det, -EPS * c / det, 1.0 / det        # i11, i12, i22


class TORAEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "tora"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        n = x.shape[0]
        q, dq, theta, dtheta = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        i11, i12, _ = _inv_inertia(theta)
        # M(theta) [qddot; thetaddot] = [-q + eps*dtheta^2*sin(theta) ; u]
        rhs0 = -q + EPS * dtheta ** 2 * torch.sin(theta)
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = dq
        f[:, 1] = i11 * rhs0
        f[:, 2] = dtheta
        f[:, 3] = i12 * rhs0
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        _, i12, i22 = _inv_inertia(x[:, 2])
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        # B = M(theta)^-1 [0; 1] -- state dependent through eps*cos(theta), but
        # ||B|| varies only 6% because det stays near 1.
        B[:, 1, 0] = i12
        B[:, 3, 0] = i22
        return B

    def _B_null_logic(self, x):
        """Null space of B^T. B occupies rows 1 and 3, so the default (identity
        on the leading x_dim - u_dim rows) would not be orthogonal to it."""
        n = x.shape[0]
        _, i12, i22 = _inv_inertia(x[:, 2])
        Bbot = self._zeros((n, self.num_dim_x, self.num_dim_x - self.num_dim_control), x)
        Bbot[:, 0, 0] = 1.0                    # q
        Bbot[:, 2, 1] = 1.0                    # theta
        Bbot[:, 1, 2] = i22                    # (0, i22, 0, -i12) . (0, i12, 0, i22) = 0
        Bbot[:, 3, 2] = -i12
        return Bbot

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        # The origin is an equilibrium and the spring is a restoring force, so
        # no trim is needed here -- unlike ball_and_beam / two_link_arm / pvtol,
        # whose references free-fall out of the box without one. The spring is
        # undamped though, so the excitation is kept small enough that the
        # oscillation it pumps in stays inside the state box.
        xr = infos.get("xref_t")
        if xr is not None:
            uref[:, 0] = -0.5 * xr[:, 3]      # light rotor damping, keeps it bounded
        for i, freq in enumerate(freqs):
            weight = weights[:, i, :]
            uref[:, 0] += weight[:, 0] * math.sin(freq * _t / self.time_bound * 2 * math.pi)
        if add_noise:
            uref += torch.randn_like(uref) * torch.abs(0.1 * uref)
        return torch.clamp(uref, self.UREF_MIN, self.UREF_MAX)

    def system_reset(self, env_ids: torch.Tensor):
        xref_0, xe_0, x_0 = self.define_initial_state(env_ids)
        freqs = list(range(1, 6))
        n = len(env_ids)
        weights = torch.randn(n, len(freqs), len(UREF_MIN), device=self.device)
        weights = 0.1 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
