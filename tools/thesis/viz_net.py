from omni.isaac.lab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab_tasks.direct.robomaster.network.actor_critic_network import ActorCriticNetwork

import os
import torch
import argparse

class WrappedActorCriticNetwork(torch.nn.Module):
    def __init__(self, original_network):
        super(WrappedActorCriticNetwork, self).__init__()
        self.original_network = original_network

    def forward(self, observation):
        mu, sigma, value, _ = self.original_network.forward(observation)
        return mu, sigma, value

# Set up argument parser
parser = argparse.ArgumentParser(description='Load a neural network from a log directory.')
parser.add_argument('--path', type=str, help='Path to the log directory containing the neural network')

# Parse arguments
args = parser.parse_args()

agent_cfg_path = os.path.join(args.path, 'params', 'agent.yaml')
agent_cfg = load_yaml(agent_cfg_path)
net_cfg = agent_cfg["params"]['network'] 

env_cfg_path = os.path.join(args.path, 'params', 'env.yaml')
env_cfg = load_yaml(env_cfg_path)

input_shape = env_cfg['observation_space']

for key, value in input_shape.items():
    if isinstance(value, list):
        input_shape[key] = tuple(value)
    else:
        input_shape[key] = (value,)

kwargs = {
        'actions_num':  env_cfg['action_space'],
        'input_shape': env_cfg['observation_space']
}

original_network = ActorCriticNetwork(net_cfg, **kwargs)
network = WrappedActorCriticNetwork(original_network)


input = {}

for key, value in input_shape.items():
        input[key] = torch.randn(1, *value)

input = {'obs': input}

network(input)

onnx_programm = torch.onnx.dynamo_export(network, input)

onnx_programm.save(os.path.join(args.path, 'model.onnx'))
print(f"Model saved to {os.path.join(args.path, 'model.onnx')}")

