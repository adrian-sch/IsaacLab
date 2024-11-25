import torch
from torch import nn
import torch.nn.functional as F

from rl_games.algos_torch.network_builder import NetworkBuilder

class TestNet(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
        num_inputs = 0

        num_inputs = input_shape[0]

        # assert(type(input_shape) is dict)
        # for k,v in input_shape.items():
        #     num_inputs +=v[0]

        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)
        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.mean_linear = nn.Linear(64, actions_num)
        self.value_linear = nn.Linear(64, self.value_size)


        self._sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)


    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu = self.mean_linear(x)
        sigma = self._sigma
        value = self.value_linear(x)
        
        # Has to retutn (mu, logstd, value, states) 
        # states is for RNN so we can ignore it here
        return mu, sigma, value, None
    




class TestNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TestNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)



class TestNetWithAuxLoss(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = 0

        self.target_key = 'aux_target'
        assert(type(input_shape) is dict)
        for k,v in input_shape.items():
            if self.target_key == k:
                self.target_shape = v[0]
            else:
                num_inputs +=v[0]

        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)
        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.mean_linear = nn.Linear(64, actions_num)
        self.value_linear = nn.Linear(64, 1)
        self.aux_loss_linear = nn.Linear(64, self.target_shape)
        
        self.aux_loss_map = {
            'aux_dist_loss' : None
        }
    def is_rnn(self):
        return False

    def get_aux_loss(self):
        return self.aux_loss_map

    def forward(self, obs):
        obs = obs['obs']
        target_obs = obs[self.target_key]
        obs = torch.cat([obs['pos'], obs['info']], axis=-1)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        action = self.mean_linear(x)
        value = self.value_linear(x)
        y = self.aux_loss_linear(x)
        self.aux_loss_map['aux_dist_loss'] = torch.nn.functional.mse_loss(y, target_obs)
        if self.central_value:
            return value, None
        return action, value, None

class TestNetAuxLossBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TestNetWithAuxLoss(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)