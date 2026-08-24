import gymnasium as gym

from . import agents

gym.register(
    id="Quadruped-PathTracking-v0",
    entry_point=f"{__name__}.env:QuadrupedPathTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.env_cfg:QuadrupedPathTrackingEnvCfg",
        "skrl_cfg_entry_point":     f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point":   f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point":    f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_c3m_cfg_entry_point":   f"{agents.__name__}:skrl_c3m_cfg.yaml",
        "skrl_sdlqr_cfg_entry_point": f"{agents.__name__}:skrl_sdlqr_cfg.yaml",
        "skrl_lqr_cfg_entry_point":   f"{agents.__name__}:skrl_lqr_cfg.yaml",
        "skrl_c2rl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_ppo_cfg.yaml",
        "skrl_c2rl_sac_cfg_entry_point": f"{agents.__name__}:skrl_c2rl_sac_cfg.yaml",
    },
)
