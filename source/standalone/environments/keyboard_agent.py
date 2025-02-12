# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from pynput import keyboard

value = 0.5

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    env_kwargs = {
        "train": False,
        "debug": True,
    }
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, **env_kwargs)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    if env.action_space.shape[1] != 3:
        raise ValueError("This script only supports environments with action space == 3.")
        
    # Initialize action value
    action_value = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    # Function to update action value based on keyboard input
    def on_press(key):
        global value
        try:
            if key.char == 'w':
                action_value[:, 0] = value
            elif key.char == 's':
                action_value[:, 0] = -value
            elif key.char == 'a':
                action_value[:, 1] = value
            elif key.char == 'd':
                action_value[:, 1] = -value
            elif key.char == 'q':
                action_value[:, 2] = value
            elif key.char == 'e':
                action_value[:, 2] = -value
            elif key.char == 'r':
                value = min(value + 0.1, 1.0)
                print(f"Value updated to: {value}")
            elif key.char == 'f':
                value = max(value - 0.1, 0.1)
                print(f"Value updated to: {value}")
        except AttributeError:
            pass
        # print(f"Action value updated to: {action_value}")

    def on_release(key):
        try:
            if key.char == 'w' or key.char == 's':
                action_value[:, 0] = 0.0
            elif key.char == 'a' or key.char == 'd':
                action_value[:, 1] = 0.0
            elif key.char == 'q' or key.char == 'e':
                action_value[:, 2] = 0.0
        except AttributeError:
            pass
        # print(f"Action value updated to: {action_value}")
    # Register keyboard events
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            print(f"Action value: {action_value}")
            env.step(action_value)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
