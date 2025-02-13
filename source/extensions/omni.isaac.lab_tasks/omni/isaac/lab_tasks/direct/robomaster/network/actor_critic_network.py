from .base_network import BaseNetwork
from .util import *
from .layer.action_holonomic_limiter import ActionHolonomicLimiter
from torch import nn
import torch
from copy import deepcopy

class ActorCriticNetwork(BaseNetwork):
    # def __init__(self, input_shape: {str, (int, ...)}, output_shape: (int,), params: {}):
    def __init__(self, params, **kwargs):
        BaseNetwork.__init__(self)

        output_shape = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')

        self._fixed_sigma = params.get("fixed_sigma", True)
        self._device = params.get("device", "cuda")

        self._actor_cnn = nn.Sequential()
        self._actor_mlp_lidar = nn.Sequential()
        self._actor_mlp_sensor = nn.Sequential()
        self._actor_mlp = nn.Sequential()

        self._actor_limiter = ActionHolonomicLimiter(**params["action_limiter"]) if "action_limiter" in params else None

        self._critic_cnn = nn.Sequential()
        self._critic_mlp_lidar = nn.Sequential()
        self._critic_mlp_sensor = nn.Sequential()
        self._critic_mlp = nn.Sequential()

        input_shape_lidar, input_shape_sensor, input_shape_goal = split_input_shape(input_shape=input_shape)
        input_shape_lidar = input_shape_lidar["lidar"]
        input_shape_sensor = input_shape_calculate_flatten_size(input_shape_sensor)
        input_shape_goal = input_shape_calculate_flatten_size(input_shape_goal)

        if "cnn" in params:
            self._actor_cnn = self._build_conv(input_shape_lidar, **params["cnn"])
            self._critic_cnn = self._build_conv(input_shape_lidar, **params["cnn"])

        output_shape_cnn = output_shape_calculate_from_model_flatten(input_shape_lidar, self._actor_cnn)

        if "mlp_lidar" in params:
            self._actor_mlp_lidar = self._build_mlp(input_size=output_shape_cnn, **params["mlp_lidar"])
            self._critic_mlp_lidar = self._build_mlp(input_size=output_shape_cnn, **params["mlp_lidar"])

        output_shape_mlp_lidar = output_shape_calculate_from_model_flatten((output_shape_cnn,), self._actor_mlp_lidar)

        if "mlp_sensor" in params:
            input_shape_c_mlp_sensor = input_shape_sensor + input_shape_goal

            self._actor_mlp_sensor = self._build_mlp(input_size=input_shape_sensor, **params["mlp_sensor"])
            self._critic_mlp_sensor = self._build_mlp(input_size=input_shape_c_mlp_sensor, **params["mlp_sensor"])

        output_shape_a_mlp_sensor = output_shape_calculate_from_model_flatten((input_shape_sensor,), self._actor_mlp_sensor)
        output_shape_c_mlp_sensor = output_shape_calculate_from_model_flatten((input_shape_c_mlp_sensor,), self._critic_mlp_sensor)

        if "mlp" in params:
            input_shape_a_mlp = output_shape_mlp_lidar + output_shape_a_mlp_sensor
            input_shape_c_mlp = output_shape_mlp_lidar + output_shape_c_mlp_sensor

            self._actor_mlp = self._build_mlp(input_size=input_shape_a_mlp, **params["mlp"])
            self._critic_mlp = self._build_mlp(input_size=input_shape_c_mlp, **params["mlp"])

        output_shape_a_mlp = output_shape_calculate_from_model_flatten((input_shape_a_mlp,), self._actor_mlp)
        output_shape_c_mlp = output_shape_calculate_from_model_flatten((input_shape_c_mlp,), self._critic_mlp)

        self._mu = nn.Linear(in_features=output_shape_a_mlp, out_features=output_shape)
        self._value = nn.Linear(in_features=output_shape_c_mlp, out_features=1)

        if self._fixed_sigma:
            self._sigma = nn.Parameter(torch.zeros(output_shape, requires_grad=True, dtype=torch.float32), requires_grad=True)
        else:
            self._sigma = nn.Linear(in_features=output_shape_a_mlp, out_features=output_shape)

        self.to(self._device)

    def forward(self, observation):
        lidar, sensor, goal = split_observation(observation['obs'])
        lidar = lidar["lidar"].to(self._device)
        sensor = torch.cat([x.flatten(start_dim=1) for x in sensor.values()], dim=1).to(self._device)

        if len(goal) > 0:
            goal = torch.cat([x.flatten(start_dim=1) for x in goal.values()], dim=1).to(self._device)
            c_in = torch.cat([sensor, goal], dim=1)
        else:
            c_in = sensor

        a_lidar_out = self._actor_cnn(lidar)
        c_lidar_out = self._critic_cnn(lidar)

        a_lidar_out = torch.flatten(a_lidar_out, start_dim=1)
        c_lidar_out = torch.flatten(c_lidar_out, start_dim=1)

        a_lidar_out = self._actor_mlp_lidar(a_lidar_out)
        c_lidar_out = self._critic_mlp_lidar(c_lidar_out)

        a_mlp_out = self._actor_mlp_sensor(sensor)
        c_mpl_out = self._critic_mlp_sensor(c_in)

        a_out = torch.cat([a_lidar_out, a_mlp_out], dim=1)
        c_out = torch.cat([c_lidar_out, c_mpl_out], dim=1)

        a_out = self._actor_mlp(a_out)
        c_out = self._critic_mlp(c_out)

        mu = torch.tanh(self._mu(a_out))
        value = self._value(c_out)

        if self._actor_limiter:
            mu = self._actor_limiter(mu)

        if self._fixed_sigma:
            sigma = mu * 0.0 + self._sigma
        else:
            sigma = self._sigma(a_out)

        # Ensure sigma is always positive
        sigma = torch.clamp(sigma, min=1e-6)

        return mu, sigma, value, None

    def act(self, observation: {str, torch.Tensor}):
        # Check for NaNs in the observation dictionary
        for key, value in observation.items():
            if torch.isnan(value).any():
                raise ValueError(f"Input tensor for key '{key}' contains NaNs")

        mu, sigma, value = self.forward(observation)
        dist = torch.distributions.Normal(mu, sigma.exp(), validate_args=False)
        select_action = dist.sample()
        select_action = torch.clamp(select_action, -1.0, 1.0)
        neglog = -dist.log_prob(select_action)

        return select_action, value, neglog.sum(dim=1, keepdims=True)


    def evaluate(self, observation: {str, torch.Tensor}, action: torch.Tensor):
        mu, sigma, value = self.forward(observation)
        dist = torch.distributions.Normal(mu, sigma.exp(), validate_args=False)
        neglog = - dist.log_prob(action.to(self._device))
        return value, neglog.sum(dim=1, keepdims=True)

    def sample(self, observation: {str, torch.Tensor}):
        mu, _, _ = self.forward(observation)
        return mu

    def device(self) -> str:
        return self._device

    @torch.jit.export
    def create_inference(self):
        inference = ActorCriticInference(self)
        inference.eval()
        return torch.jit.script(inference)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def is_rnn(self):
        return False


class ActorCriticInference(nn.Module):
    def __init__(self, network: ActorCriticNetwork):
        nn.Module.__init__(self)

        self._cnn = deepcopy(network._actor_cnn)
        self._mlp_lidar = deepcopy(network._actor_mlp_lidar)
        self._mlp_sensor = deepcopy(network._actor_mlp_sensor)
        self._mlp = deepcopy(network._actor_mlp)
        self._mu = deepcopy(network._mu)
        self._device = deepcopy(network._device)

        self.to(self._device)

    def forward(self, lidar, sensor):
        lidar = self._cnn(lidar)
        lidar = torch.flatten(lidar, start_dim=1)
        lidar = self._mlp_lidar(lidar)

        sensor = self._mlp_sensor(sensor)

        out = torch.cat([lidar, sensor], dim=1)
        out = self._mlp(out)

        action = torch.tanh(self._mu(out))

        return action
    
# wrappter for exporting for Netron without None in return
class ActorCriticVizualization(nn.Module):
    def __init__(self, network: ActorCriticNetwork):
        nn.Module.__init__(self)
        self.network = network

    def forward(self, observation):
        mu, sigma, value, _ = self.network.forward(observation)
        return mu, sigma, value
