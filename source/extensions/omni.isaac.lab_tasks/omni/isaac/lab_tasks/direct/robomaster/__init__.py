# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .robomaster_env import RobomasterEnv, RobomasterEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Robomaster-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.robomaster:RobomasterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RobomasterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
