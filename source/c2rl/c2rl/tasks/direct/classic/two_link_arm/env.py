"""Two-link planar manipulator tracking environment (batched PyTorch).

Reference
---------
M. W. Spong, S. Hutchinson and M. Vidyasagar, "Robot Modeling and Control",
Wiley, 2006, Ch. 7 (Euler-Lagrange equations for the two-link planar elbow arm).
The inertia, Coriolis and gravity terms below are that standard model.

Why this plant is here
----------------------
category b, mechanical/inertial variation -- and the canonical one.  With both
joints actuated the input matrix is exactly the inverse inertia,

    B(q) = M(q)^-1,     M11 depends on cos(q2)

so the arm is hardest to accelerate fully extended (largest inertia) and the
conditioning of B degrades toward the kinematic singularity q2 -> 0.  Measured
sv(B) spread 5.753 over the state box below, the largest of any mechanical plant
in this repo (cartpole 2.213, ball-and-beam 3.470).

Contrast with the acrobot (1.454), which shares this M(q): projecting M^-1 onto
a single actuated column washes out most of the conditioning, so underactuation
does not by itself produce a state-dependent sigma(B).
"""

from __future__ import annotations

import math

import torch

from ..common.env_base import BaseEnv

# Two-Link arm parameters (Spong et al. 2006, standard elbow-arm values)
M1, M2 = 1.0, 1.0          # kg
L1, L2 = 1.0, 1.0          # m, link lengths
LC1, LC2 = 0.5, 0.5        # m, distance to each link's centre of mass
I1, I2 = 0.083, 0.083      # kg m^2, link inertias about their own COM
G = 9.81

STATE_NAMES = ("joint_pos_0", "joint_pos_1", "joint_vel_0", "joint_vel_1")

# X bounds. q2 is kept off the exact singularity |q2| = pi (fully folded), where
# M(q) is worst-conditioned and B blows up; +-2.9 rad leaves a small margin.
X_MIN = [-math.pi, -2.9, -3.0, -3.0]
X_MAX = [math.pi, 2.9, 3.0, 3.0]

# Episode ends the first step x leaves this box (opt-in: --terminate_out_of_box).
# Defaults to the state box itself, i.e. it fires exactly where env_base.step's
# clamp already silently pins a diverged env -- the same event, reported instead
# of hidden. Tighten it here to end failing episodes sooner.
X_TERMINATION_MIN = list(X_MIN)
X_TERMINATION_MAX = list(X_MAX)

XREF_INIT_MIN = [-0.5, -0.5, -0.3, -0.3]
XREF_INIT_MAX = [0.5, 0.5, 0.3, 0.3]

XE_INIT_MIN = [-0.3, -0.3, -0.3, -0.3]
XE_INIT_MAX = [0.3, 0.3, 0.3, 0.3]

lim = 1.0
XE_MIN = [-lim, -lim, -lim, -lim]
XE_MAX = [lim, lim, lim, lim]

# reference control bounds -- joint torques (N m). The applied box is 2x this.
# Holding the arm horizontal against gravity needs
# (m1*lc1 + m2*l1)*g + m2*lc2*g = 14.7 + 4.9 N m at the shoulder, so +-10
# reference / +-20 applied covers the static load.
# Raised from +-10: holding the arm horizontal needs 19.6 N m at the shoulder,
# so the gravity trim itself was being clipped and the reference fell anyway.
UREF_MIN = [-30.0, -30.0]
UREF_MAX = [30.0, 30.0]


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
    "dt": 0.01,
    "time_bound": 5.0,
    "q": 1.0,
    "r": 0.0,
}


def _inertia(q2):
    """M(q) entries. Only q2 enters -- the shoulder angle does not change the
    inertia seen at the joints, which is why sigma(B) varies with q2 alone."""
    c2 = torch.cos(q2)
    m11 = M1 * LC1 ** 2 + M2 * (L1 ** 2 + LC2 ** 2 + 2.0 * L1 * LC2 * c2) + I1 + I2
    m12 = M2 * (LC2 ** 2 + L1 * LC2 * c2) + I2
    m22 = torch.full_like(q2, M2 * LC2 ** 2 + I2)
    return m11, m12, m22


def _inv_inertia(q2):
    """M(q)^-1 in closed form -- a 2x2 inverse, no linear solve per state."""
    m11, m12, m22 = _inertia(q2)
    det = m11 * m22 - m12 * m12
    return m22 / det, -m12 / det, m11 / det


class TwoLinkArmEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        sample_mode: str = "uniform",
        time_bound: float | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        self.task = "two_link_arm"
        super().__init__(
            self._build_cfg(ENV_CONFIG, sample_mode=sample_mode, time_bound=time_bound, dt=dt, **kwargs),
            num_envs=num_envs,
            device=device,
        )

    def _f_logic(self, x):
        n = x.shape[0]
        q1, q2, dq1, dq2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        s2 = torch.sin(q2)
        # Coriolis/centrifugal (Spong eq. 7.__): h = -m2 l1 lc2 sin(q2)
        h = -M2 * L1 * LC2 * s2
        c1 = h * dq2 * dq1 + h * (dq1 + dq2) * dq2
        c2_ = -h * dq1 * dq1
        # gravity
        g1 = (M1 * LC1 + M2 * L1) * G * torch.cos(q1) + M2 * LC2 * G * torch.cos(q1 + q2)
        g2 = M2 * LC2 * G * torch.cos(q1 + q2)
        i11, i12, i22 = _inv_inertia(q2)
        rhs1, rhs2 = -(c1 + g1), -(c2_ + g2)
        f = self._zeros((n, self.num_dim_x), x)
        f[:, 0] = dq1
        f[:, 1] = dq2
        f[:, 2] = i11 * rhs1 + i12 * rhs2
        f[:, 3] = i12 * rhs1 + i22 * rhs2
        return f

    def _B_logic(self, x):
        n = x.shape[0]
        i11, i12, i22 = _inv_inertia(x[:, 1])
        B = self._zeros((n, self.num_dim_x, self.num_dim_control), x)
        # B = M(q)^-1 exactly -- the plant's whole reason for being here.
        B[:, 2, 0] = i11
        B[:, 2, 1] = i12
        B[:, 3, 0] = i12
        B[:, 3, 1] = i22
        return B

    def sample_reference_controls(self, freqs, weights, _t, infos, add_noise=False):
        n = weights.shape[0]
        uref = torch.zeros(n, self.num_dim_control, device=self.device)
        # Trim: G(q), the torque that holds the arm against gravity at the
        # Current reference state. A manipulator reference generated from
        # sinusoids around zero torque is a falling arm -- it clamped its joint
        # velocities 22% of the time regardless of amplitude, and the tracking
        # controller then saturated 75-100% of the time chasing it.
        xr = infos.get("xref_t")
        if xr is not None:
            q1, q2 = xr[:, 0], xr[:, 1]
            uref[:, 0] = ((M1 * LC1 + M2 * L1) * G * torch.cos(q1)
                          + M2 * LC2 * G * torch.cos(q1 + q2))
            uref[:, 1] = M2 * LC2 * G * torch.cos(q1 + q2)
            # PD about the start configuration. Gravity compensation alone
            # leaves a double integrator: xref_0 has nonzero joint velocities, so
            # the reference coasts until it clamps. The plant is open-loop
            # unstable, so a bounded reference has to be generated by a
            # stabilising feedback -- open-loop excitation cannot produce one.
            # (xref, uref) stays an exact trajectory either way: it is the clamp
            # that breaks the pair, and this is what stops the clamp firing.
            q0 = infos["xref_0"]
            uref[:, 0] += -8.0 * (q1 - q0[:, 0]) - 5.0 * xr[:, 2]
            uref[:, 1] += -8.0 * (q2 - q0[:, 1]) - 5.0 * xr[:, 3]
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
        weights = 2.0 * weights / torch.sqrt((weights ** 2).sum(dim=1, keepdim=True))
        xref_arr, uref_arr, length = self._rollout_reference(xref_0, freqs, weights)
        return x_0, xref_arr, uref_arr, length
