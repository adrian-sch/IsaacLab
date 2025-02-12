from omni.isaac.lab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from omni.isaac.lab.utils.io import load_yaml
from omni.isaac.lab_tasks.direct.robomaster.network.actor_critic_network import ActorCriticNetwork, ActorCriticInference

import os
import torch
import argparse
import pickle

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

script_dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(script_dir, 'sample_inputs.pkl')

with open(pickle_path, 'rb') as f:
    sample_input = pickle.load(f)
    
print(sample_input)


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

original_network: torch.nn.Module = ActorCriticNetwork(net_cfg, **kwargs)
original_network.load_state_dict(loaded_model['model'], strict=False)
network = ActorCriticInference(original_network)

print(input_shape)
print(sample_input)

sample_input_tensor = {
        'lidar': torch.tensor(sample_input['l_lidar_'], device='cuda:0'),
        'sensor': torch.tensor(sample_input['l_sensor_'], device='cuda:0')
}

# input = {}

# for key, value in input_shape.items():
#         if key == "goal":
#                 continue
#         input[key] = torch.randn(1, *value, device='cuda:0')
#         print("input size ", key, input[key].size())
#         print("sample size ", key, sample_input_tensor[key].size())

pytorch_out = network(sample_input_tensor['lidar'], sample_input_tensor['sensor'])

onnx_programm = torch.onnx.dynamo_export(network, sample_input_tensor['lidar'], sample_input_tensor['sensor'])

onnx_out = onnx_programm(sample_input_tensor['lidar'], sample_input_tensor['sensor'])

print("Pytorch out:", pytorch_out)
print("Onnx out:", onnx_out)

print("Diff: ", pytorch_out - torch.tensor(onnx_out, device='cuda:0'))


