"""Classic Car tracking environment, v1: the low-speed velocity box (class III).

Registered as ``classic-car-v1``, a VERSION of the car rather than a separate
task, because it is the same plant — see ``env.py`` for why the box, and only
the box, differs from ``classic-car-v0``, and why that alone turns the certified
contraction rate from a constant into a function of the state.

Only ``c2rl-ppo`` gets its own config here, because it is the only algorithm with
a state-box-dependent artifact (the offline ``{x -> W*(x)}`` CM dataset, whose
cache key does not include the box). Every other algorithm reads the car's own
config unchanged, since nothing in those files depends on the box.
"""

import gymnasium as gym

from ..car import agents as car_agents
from . import agents

gym.register(
    id="classic-car-v1",
    entry_point=f"{__name__}.env:CarV1Env",
    disable_env_checker=True,
    kwargs={
        "skrl_cfg_entry_point":       f"{car_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point":   f"{car_agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point":   f"{car_agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_c3m_cfg_entry_point":   f"{car_agents.__name__}:skrl_c3m_cfg.yaml",
        "skrl_lqr_cfg_entry_point":   f"{car_agents.__name__}:skrl_lqr_cfg.yaml",
        "skrl_sdlqr_cfg_entry_point": f"{car_agents.__name__}:skrl_sdlqr_cfg.yaml",
        "skrl_cvstem_lqr_cfg_entry_point": f"{car_agents.__name__}:skrl_cvstem_lqr_cfg.yaml",
        "skrl_c2rl_sac_cfg_entry_point": f"{car_agents.__name__}:skrl_c2rl_sac_cfg.yaml",
        # The one that must not be shared: its cm_data_path points at
        # data/classic/car_v1/, not data/classic/car/.
        "skrl_c2rl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_ppo_cfg.yaml",
    },
)
