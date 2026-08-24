import gymnasium as gym

from . import agents

gym.register(
    id="Quadruped-VelTracking-v0",
    entry_point=f"{__name__}.env:QuadrupedVelTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:QuadrupedVelTrackingEnvCfg",
        "skrl_cfg_entry_point":     f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point":   f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
