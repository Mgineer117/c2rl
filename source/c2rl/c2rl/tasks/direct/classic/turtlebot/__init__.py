import gymnasium as gym

gym.register(
    id="classic-turtlebot-v0",
    entry_point=f"{__name__}.env:TurtlebotEnv",
    disable_env_checker=True,
    kwargs={
        "skrl_c3m_cfg_entry_point": f"{__name__}.agents:skrl_c3m_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{__name__}.agents:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{__name__}.agents:skrl_sac_cfg.yaml",
        "skrl_c2rl_ppo_cfg_entry_point": f"{__name__}.agents:skrl_c2rl_ppo_cfg.yaml",
        "skrl_c2rl_sac_cfg_entry_point": f"{__name__}.agents:skrl_c2rl_sac_cfg.yaml",
        "skrl_lqr_cfg_entry_point": f"{__name__}.agents:skrl_lqr_cfg.yaml",
        "skrl_sdlqr_cfg_entry_point": f"{__name__}.agents:skrl_sdlqr_cfg.yaml",
        "skrl_cvstem_lqr_cfg_entry_point": f"{__name__}.agents:skrl_cvstem_lqr_cfg.yaml",
    },
)
