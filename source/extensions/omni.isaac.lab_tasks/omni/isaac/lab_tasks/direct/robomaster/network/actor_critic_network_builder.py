from .actor_critic_network import ActorCriticNetwork
from rl_games.algos_torch.network_builder import NetworkBuilder

class ActorCriticNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return ActorCriticNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

