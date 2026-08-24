import gymnasium as gym

from . import agents

gym.register(
    id="Cartpole-v0",
    entry_point=f"{__name__}.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "skrl_cfg_entry_point":     f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point":   f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
