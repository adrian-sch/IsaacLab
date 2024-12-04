from .base_network import BaseNetwork
from .util import input_shape_split_into_lidar_and_sensor, input_shape_calculate_flatten_size, output_shape_calculate_from_model_flatten, observation_split_into_lidar_and_sensor
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



        self._separate = params.get("separate", False)
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

        input_shape_lidar, input_shape_sensor = input_shape_split_into_lidar_and_sensor(input_shape=input_shape)
        input_shape_lidar = input_shape_lidar["lidar"]
        input_shape_sensor = input_shape_calculate_flatten_size(input_shape_sensor)

        if "cnn" in params:
            self._actor_cnn = self._build_conv(input_shape_lidar, **params["cnn"])
            if self._separate:
                self._critic_cnn = self._build_conv(input_shape_lidar, **params["cnn"])

        output_shape_cnn = output_shape_calculate_from_model_flatten(input_shape_lidar, self._actor_cnn)

        if "mlp_lidar" in params:
            self._actor_mlp_lidar = self._build_mlp(input_size=output_shape_cnn, **params["mlp_lidar"])
            if self._separate:
                self._critic_mlp_lidar = self._build_mlp(input_size=output_shape_cnn, **params["mlp_lidar"])

        output_shape_mlp_lidar = output_shape_calculate_from_model_flatten((output_shape_cnn,), self._actor_mlp_lidar)

        if "mlp_sensor" in params:
            self._actor_mlp_sensor = self._build_mlp(input_size=input_shape_sensor, **params["mlp_sensor"])
            if self._separate:
                self._critic_mlp_sensor = self._build_mlp(input_size=input_shape_sensor, **params["mlp_sensor"])

        output_shape_mlp_sensor = output_shape_calculate_from_model_flatten((input_shape_sensor,), self._actor_mlp_sensor)
        input_shape_mlp = output_shape_mlp_lidar + output_shape_mlp_sensor

        if "mlp" in params:
            self._actor_mlp = self._build_mlp(input_size=input_shape_mlp, **params["mlp"])
            if self._separate:
                self._critic_mlp = self._build_mlp(input_size=input_shape_mlp, **params["mlp"])

        output_shape_mlp = output_shape_calculate_from_model_flatten((input_shape_mlp,), self._actor_mlp)

        self._value = nn.Linear(in_features=output_shape_mlp, out_features=1)
        self._mu = nn.Linear(in_features=output_shape_mlp, out_features=output_shape)

        if self._fixed_sigma:
            self._sigma = nn.Parameter(torch.zeros(output_shape, requires_grad=True, dtype=torch.float32), requires_grad=True)
        else:
            self._sigma = nn.Linear(in_features=output_shape_mlp, out_features=output_shape)

        self.to(self._device)

    def forward(self, observation):
        lidar, sensor = observation_split_into_lidar_and_sensor(observation['obs'])
        lidar = lidar["lidar"].to(self._device)
        sensor = torch.cat([x.flatten(start_dim=1) for x in sensor.values()], dim=1).to(self._device)

        sensor += torch.normal(0, 0.0125, sensor.shape, device=self._device)

        if self._separate:
            a_lidar = self._actor_cnn(lidar)
            c_lidar = self._critic_cnn(lidar)

            a_lidar = torch.flatten(a_lidar, start_dim=1)
            c_lidar = torch.flatten(c_lidar, start_dim=1)

            a_lidar = self._actor_mlp_lidar(a_lidar)
            c_lidar = self._critic_mlp_lidar(c_lidar)

            a_sensor = self._actor_mlp_sensor(sensor)
            c_sensor = self._critic_mlp_sensor(sensor)

            a_out = torch.cat([a_lidar, a_sensor], dim=1)
            c_out = torch.cat([c_lidar, c_sensor], dim=1)

            a_out = self._actor_mlp(a_out)
            c_out = self._critic_mlp(c_out)

            mu = torch.tanh(self._mu(a_out))
            value = self._value(c_out)

        else:
            a_lidar = self._actor_cnn(lidar)
            a_lidar = torch.flatten(a_lidar, start_dim=1)
            a_lidar = self._actor_mlp_lidar(a_lidar)

            a_sensor = self._actor_mlp_sensor(sensor)

            a_out = torch.cat([a_lidar, a_sensor], dim=1)
            a_out = self._actor_mlp(a_out)

            mu = torch.tanh(self._mu(a_out))
            value = self._value(a_out)

        if self._actor_limiter:
            mu = self._actor_limiter(mu)

        if self._fixed_sigma:
            sigma = mu * 0.0 + self._sigma
        else:
            sigma = self._sigma(a_out)

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
