"""Ball-and-beam tracking environment (batched PyTorch).

Reference
---------
J. Hauser, S. Sastry and P. Kokotovic, "Nonlinear control via approximate
input-output linearization: the ball and beam example", IEEE Transactions on
Automatic Control, 37(3):392-398, 1992.  The classic benchmark for a system
whose exact input-output linearization is singular; the parameter values and
the (r, rdot, theta, thetadot) formulation below follow that paper.

Why this plant is here
----------------------
category b, mechanical/inertial variation.  The beam's moment of inertia carries
the ball's contribution ``m * r^2``, so the torque-to-angular-acceleration gain

    B[thetadot, tau] = 1 / (J_beam + m * r^2)

falls as the ball rolls outward.  Measured sv(B) spread 3.470 over the state box
below -- larger than cartpole's 2.213, which is the repo's previous maximum.
That is what lets a state subset certify a faster contraction rate than the whole
box: B enters the CV-STEM LMI only through ``nu*(2/r)*B B^T`` with ``nu`` shared
across states, so a state with less authority cannot be rescued by any metric,
only excluded.
"""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# Ball-And-Beam parameters (Hauser et al. 1992)
M_BALL = 0.05        # kg
R_BALL = 0.01        # m, ball radius
J_BALL = 2.0e-6      # kg m^2, ball moment of inertia
J_BEAM = 0.02        # kg m^2, beam moment of inertia about the pivot
G = 9.81

# The ball's translational inertia, constant: m + J_ball/R^2.
M_EFF = M_BALL + J_BALL / (R_BALL ** 2)

STATE_NAMES = ("joint_pos_0", "joint_vel_0", "pitch", "pitch_rate")
# joint_* rather than pos_x: the dynamics depend on the ball's absolute position
# along the beam (through m*r^2), so this plant is not translation invariant and
# must not be given the translation quotient.

# X bounds. r is the ball position on a 2 m beam; the beam tilt is kept inside
# +-35 deg, past which the ball leaves the rail on real hardware.
X_MIN = [-1.0, -2.0, -0.6, -2.0]
X_MAX = [1.0, 2.0, 0.6, 2.0]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-0.3, -0.2, -0.1, -0.2]
XREF_INIT_MAX = [0.3, 0.2, 0.1, 0.2]

XE_INIT_MIN = [-0.2, -0.2, -0.1, -0.2]
XE_INIT_MAX = [0.2, 0.2, 0.1, 0.2]

lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds -- beam torque (N m). The applied box is 2x this.
# Holding the ball at the beam tip against gravity needs m*g*r = 0.49 N m, so
# +-0.5 reference / +-1.0 applied covers the static requirement with headroom.
# Raised from +-0.5: that box was sized for statics only -- holding the ball
# at the beam tip alone needs m*g*r = 0.49 N m, leaving nothing for feedback,
# and the certificate had to weight the control r=102.4 to stay inside it.
# At +-2.0 the same search returns lbd 0.0780 -> 0.8889 with r 102.4 -> 25.6.
UREF_MIN = [-2.0]
UREF_MAX = [2.0]

# Initial state drawn directly from this box (see env_base.X_INIT_MIN),
# placing every episode start in the plant's low-lambda region:
# lbd*(x0) median 5.8603 -> 2.7354. The sign dims are
# mirrored by one shared sign, because the slow set is two lobes on a
# diagonal that no single box can cover.
X_INIT_MIN = [0.6876, -1.9602, -0.5876, 1.3482]
X_INIT_MAX = [0.9803, -0.0428, -0.0916, 1.9923]
X_INIT_SIGN_DIMS = [0, 1, 2, 3]

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
    "dt": 0.01,
    "time_bound": 5.0,
    "q": 1.0,
    "r": 0.0,
}


class BallAndBeamEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "ball_and_beam"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        n = x.shape[0]
        r, rdot, theta, thetadot = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = rdot
        f[:, 1] = (M_BALL * r * thetadot ** 2 - M_BALL * G * torch.sin(theta)) / M_EFF
        f[:, 2] = thetadot
        # Coriolis (-2 m r rdot thetadot) and the ball's gravity moment, both
        # divided by the r-Dependent beam inertia.
        f[:, 3] = (-2.0 * M_BALL * r * rdot * thetadot
                   - M_BALL * G * r * torch.cos(theta)) / (J_BEAM + M_BALL * r ** 2)
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        r = x[:, 0]
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        # The whole point of this plant: authority decays as 1/(J + m r^2).
        B[:, 3, 0] = 1.0 / (J_BEAM + M_BALL * r ** 2)
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        # Trim: the torque that cancels the ball's gravity moment at the current
        # reference state, so the unforced reference holds its tilt instead of
        # collapsing into the box. Without it the reference clamps 64% of the
        # time at any excitation amplitude -- the drift, not the excitation, was
        # driving it out.
        xr = infos.get("xref_t")
        if xr is not None:
            r_ball, theta = xr[:, 0], xr[:, 2]
            uref[:, 0] = (M_BALL * G * r_ball * torch.cos(theta)
                          + 2.0 * M_BALL * r_ball * xr[:, 1] * xr[:, 3])
            # The ball is open-Loop unstable on the beam: any tilt accelerates
            # it outward until r clamps (measured 62% of steps, at every
            # excitation amplitude). Cancelling the beam's own gravity moment
            # does not help -- it is the ball that runs away. A bounded
            # reference therefore needs the tilt commanded by a stabilising
            # feedback on r, which is what a real ball-and-beam controller does.
            # rddot ~= -(m*g/M_EFF)*theta = -0.097*theta, so driving r -> 0 needs
            # theta commanded with the same sign as r, not the opposite. The
            # 1/0.097 = 10.3 factor is why the tilt command is clamped: the
            # cascade asks for far more tilt than the beam has.
            theta_cmd = (3.1 * r_ball + 5.1 * xr[:, 1]).clamp(-0.45, 0.45)
            uref[:, 0] += (J_BEAM + M_BALL * r_ball ** 2) * (
                -25.0 * (theta - theta_cmd) - 8.0 * xr[:, 3])
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
        # excitation sized against the box: halving it takes reference clamping 6.6% -> 0.9%.
        weights = 0.075 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
