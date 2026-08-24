"""The car at low speed — v1 of the car, where the contraction rate stops being
constant.

``classic-car-v0`` and ``classic-car-v1`` are the SAME vehicle: same ``f``, same
``B``, same reward, same reference generator. They are two versions of one env
rather than two envs because the only thing that differs is the velocity box —
and that one change is enough to move the certified contraction rate from a
constant to a function of the state.

``docs/dynamics_taxonomy.md`` §2.2 gives the car's Hautus margin at ``s = 0`` as
``sigma = min(1, v)``, so the box alone decides the contraction class:

    classic-car-v0   v in [1.0, 2.0]   sigma == 1       -> class II,  lambda*(x) constant
    classic-car-v1   v in [0.2, 2.0]   sigma == v < 1   -> class III, lambda*(x) state-dependent

Measured on the shipped envelope (see figures/rate_car.svg, figures/rate_car_v1.svg):

    v0   lambda* = 4.884 everywhere
    v1   lambda* sweeps 1.15 -> 3.92 with v, saturating at 3.920 once v >= 1.10
         (past v = 1 the min in sigma stops biting and v1 IS v0)

Measured at lbd=0.3, N=200 (``scripts/feasibility_certificate.py --verify``):

    env         class  hautus     nu      rho spread  lam_C spread  ||K||max
    car          II    1.000e+00    9.536     1.0000       1.0000      2.862
    car_v1     III   2.049e-01  147.5      15.4690       7.4668      8.775

That pair is the one within-plant control available for "does the contraction
class change how RL behaves": any comparison across two different plants
confounds the class with dimension, stiffness, actuation and reward scale.

Both ends of the box move, and both are load-bearing:

* ``x_min[3]``: 1.0 -> 0.2 lets the state reach the weak-authority region at all.
* ``xref_init`` v: pinned 1.5 -> [0.3, 1.5] makes the reference visit it.
  ``sample_reference_controls`` never drives the acceleration channel, so xref
  holds its initial v for the whole episode; without widening the initial draw,
  low-v states would only ever be transient tracking error and the plant would
  spend no meaningful time where sigma < 1.
"""

from __future__ import annotations

from ..car.env import ENV_CONFIG as CAR_ENV_CONFIG
from ..car.env import STATE_NAMES, CarEnv  # noqa: F401  re-exported for parity

V_WEAK_LO = 0.2
VREF_WEAK_LO = 0.3
# Upper end of the INITIAL velocity draw, and the velocity slice of XE_INIT.
#
# Measured lambda*(v) for this env (min-projected joint CV-STEM SDP, 15 cells
# across the box, see figures/minproj_car_v1.png):
#
#     v    0.20  0.33  0.46  0.59  0.71  0.84  0.97  1.10 ... 2.00
#     l*   1.145 1.709 2.207 2.654 3.071 3.466 3.833 3.920 ... 3.920
#
# It rises monotonically and then SATURATES at 3.920 from v = 1.10 on, because
# the Hautus margin is sigma = min(1, v) and the min stops biting once v >= 1 --
# past that point car_v1 is just the car. So the low-rate region this env
# exists to study is v well under 1.
#
# With xref's velocity drawn from [0.3, 0.6] and xe's from +-0.1, x_0's velocity
# lands in [0.2, 0.7], i.e. lambda* in [1.15, 3.07]: the whole varying part of
# the curve and none of the saturated tail. It also needs no clamping, since 0.2
# is exactly the box floor -- clamping would pile a spike of probability onto the
# boundary instead of leaving a clean draw.
#
# Inherited instead, xref's velocity ran to 1.5 and xe's to +-1, so x_0 covered
# the entire [0.2, 2.0] box and most episodes began in the saturated region --
# which is the one place car_v1 carries no more information than car.
VREF_WEAK_HI = 0.6
XE_INIT_V = 0.1

# Only the bounds that define the weak-authority region move; every other key is
# the car's. Passed as constructor kwargs, which BaseEnv._build_cfg merges over
# ENV_CONFIG in place.
BOX_OVERRIDES = {
    "x_min": [*CAR_ENV_CONFIG["x_min"][:3], V_WEAK_LO],
    "xref_init_min": [*CAR_ENV_CONFIG["xref_init_min"][:3], VREF_WEAK_LO],
    "xref_init_max": [*CAR_ENV_CONFIG["xref_init_max"][:3], VREF_WEAK_HI],
    # Velocity slice only: the position/yaw perturbations stay the car's, since
    # lambda* is flat in those (measured ptp 0.0000 across yaw).
    "xe_init_min": [*CAR_ENV_CONFIG["xe_init_min"][:3], -XE_INIT_V],
    "xe_init_max": [*CAR_ENV_CONFIG["xe_init_max"][:3], XE_INIT_V],
    # The early-termination box follows this env's state box, not the car's.
    # Inherited unchanged through the {**CAR_ENV_CONFIG, ...} merge below it
    # would still start at v = 1.0 and end every episode the moment the velocity
    # drops under it -- precisely the weak-authority region (sigma = v < 1) this
    # env exists to spend time in. It would not trip BaseEnv's "inside
    # [X_MIN, X_MAX]" check either, since a box that is too narrow is legal; it
    # would just silently delete the experiment.
    "x_termination_min": [*CAR_ENV_CONFIG["x_min"][:3], V_WEAK_LO],
    "x_termination_max": list(CAR_ENV_CONFIG["x_max"]),
}

ENV_CONFIG = {**CAR_ENV_CONFIG, **BOX_OVERRIDES}


class CarV1Env(CarEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**BOX_OVERRIDES, **kwargs})
        # After super().__init__, which sets self.task = "car". Anything keyed on
        # the task name -- notably the offline CM dataset directory -- must not
        # collide with the class-II variant's.
        self.task = "car_v1"
