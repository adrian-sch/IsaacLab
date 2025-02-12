from omni.isaac.lab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab_tasks.direct.robomaster.network.actor_critic_network import ActorCriticNetwork, ActorCriticInference

import os
import torch
import argparse

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

weight_path = os.path.join(args.path, 'nn', 'robomaster_direct.pth')

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

loaded_model = torch.load(weight_path)
state_dict = loaded_model["model"]  # Extract only model weights

# Remove potential prefixes if needed
new_state_dict = {k.replace("a2c_network.", ""): v for k, v in state_dict.items()}


print(loaded_model["model"].keys())
print(new_state_dict.keys())

original_network: torch.nn.Module = ActorCriticNetwork(net_cfg, **kwargs)
missing, unexpected = original_network.load_state_dict(new_state_dict, strict=False)
network = ActorCriticInference(original_network)

if missing or unexpected:
        print("WARNING: MISSING OR UNEXPECTED KEYS, CHECK THE NETWORK, PARAMETERS MAY NOT BE LOADED CORRECTLY")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

input = {}

for key, value in input_shape.items():
        input[key] = torch.randn(1, *value, device='cuda:0')

# input = (input['lidar'], input['sensor'])

with torch.no_grad():
        print(network(input['lidar'], input['sensor']))

onnx_programm = torch.onnx.dynamo_export(network, input['lidar'], input['sensor'])

onnx_programm.save(os.path.join(args.path, 'inference_model.onnx'))
print(f"Model saved to {os.path.join(args.path, 'inference_model.onnx')}")

