"""Classic (non-Isaac) analytical tracking environments.

Each subpackage registers a gym environment so it is discoverable by
``scripts/list_envs.py`` and runnable with ``scripts/skrl/train.py --classic``.

Importing this package registers every classic env. This works both as
``c2rl.tasks.direct.classic`` (under Isaac) and as a standalone
``classic`` package on sys.path (no Isaac) — the registration
f-strings adapt to whichever package name they are imported under.
"""

import warnings

# gymnasium reads the trailing -vN as a VERSION, so registering classic-car-v1
# makes it warn "classic-car-v0 is out of date, consider upgrading to v1" on
# every single car-v0 construction. That is wrong here and actively misleading:
# v0 and v1 are peers, two velocity boxes of one plant (v0 keeps v >= 1 where the
# Hautus margin sigma = min(1, v) is pinned at 1 and lambda* is constant; v1 opens
# it to v = 0.2 where sigma = v < 1 and lambda* becomes state-dependent). v0 is
# the baseline, not the obsolete one. Narrow filter, matched on this exact
# message, so a genuine gymnasium deprecation still reaches us.
warnings.filterwarnings(
    "ignore",
    message=r".*classic-car-v0 is out of date.*",
    category=DeprecationWarning,
)

from . import car  # noqa: E402,F401  registers classic-car-v0
from . import car_v1  # noqa: F401  registers classic-car-v1 (same plant, weak-authority velocity box)
from . import cartpole  # noqa: F401 registers classic-cartpole-v0
from . import turtlebot  # noqa: F401 registers classic-turtlebot-v0
from . import segway  # noqa: F401 registers classic-segway-v0
from . import quadrotor  # noqa: F401 registers classic-quadrotor-v0
from . import ball_and_beam  # noqa: F401  registers classic-ball_and_beam-v0
from . import two_link_arm  # noqa: F401  registers classic-two_link_arm-v0
from . import aircraft  # noqa: F401  registers classic-aircraft-v0
from . import auv  # noqa: F401  registers classic-auv-v0
from . import pvtol  # noqa: F401  registers classic-pvtol-v0
from . import tora  # noqa: F401  registers classic-tora-v0

__all__ = ["car", "car_v1", "cartpole", "turtlebot", "segway", "quadrotor", "ball_and_beam", "two_link_arm", "aircraft", "auv", "pvtol", "tora"]

# Short name -> gym id, for the envs whose id is not ``classic-<name>-v0``.
# car_v1 is the same plant as car with a wider velocity box, so it registers as a
# VERSION of car rather than as a separate task -- see car_v1/env.py for why the
# box, and only the box, is what differs.
#
# One resolver, imported by everything that turns a short name into an id
# (tests/conftest.py, scripts/minproj_plot.py, scripts/list_envs.py,
# visualization/viz_common.py). Three private copies of this map is how one of
# them silently keeps building classic-car_v1-v0 after the next rename.
ENV_IDS = {"car_v1": "classic-car-v1"}


def env_id(name: str) -> str:
    """``"car"`` -> ``classic-car-v0``; ``"car_v1"`` -> ``classic-car-v1``."""
    return ENV_IDS.get(name, f"classic-{name}-v0")
