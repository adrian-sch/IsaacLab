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
from .robomaster_glide_env import RobomasterGlideEnv, RobomasterGlideEnvCfg

# register network for rl games
from .test_network import TestNetBuilder 
from rl_games.algos_torch import model_builder
model_builder.register_network('testnet', TestNetBuilder)

from .network.actor_critic_network_builder import ActorCriticNetworkBuilder
model_builder.register_network('christian_net', ActorCriticNetworkBuilder)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Robomaster-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.robomaster:RobomasterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RobomasterEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_lidar.yaml",
    },
)

gym.register(
    id="Isaac-Robomaster-Glide-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.robomaster:RobomasterGlideEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RobomasterGlideEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)