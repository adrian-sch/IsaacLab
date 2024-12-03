import torch
from torch import nn
from .layer.min_pooling import MinPool1d
from .layer.impala_sequential import ImpalaSequential1d


class BaseNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def _activation_from_name(self, activation: str = None):
        if activation == 'linear':
            return nn.Identity()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'swish':
            return nn.SiLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'softplus':
            return nn.Softplus()
        else:
            raise Exception("Activation function '" + activation + "' is unknown!")

    def _build_conv(self, input_shape, **kwargs):
        ctype = kwargs.pop("type", "conv1d")
        if ctype == "conv1d":
            return self._build_conv1d(input_shape, **kwargs)
        elif ctype == "impala1d":
            return self._build_impala1d(input_shape, **kwargs)
        elif ctype == "conv2d":
            return self._build_conv2d(input_shape, **kwargs)
        else:
            raise Exception("Unknown conv type!")

    def _build_impala1d(self, input_shape, convs, activation: str = None, min_pooling: int = -1):
        in_channels = input_shape[0]
        layers = []

        if 1 < min_pooling:
            layers.append(MinPool1d(min_pooling))
        for conv in convs:
            layers.append(ImpalaSequential1d(
                in_channels=in_channels,
                out_channels=conv['filters'],
                kernel_size=conv['kernel_size'],
                stride=conv['stride'],
                padding=conv['padding']))
            if not (activation is None):
                layers.append(self._activation_from_name(activation))
            in_channels = conv['filters']

        return nn.Sequential(*layers)


    def _build_conv1d(self, input_shape, convs, activation: str = None, min_pooling: int = -1):
        in_channels = input_shape[0]
        layers = []
        if 1 < min_pooling:
            layers.append(MinPool1d(min_pooling))
        for conv in convs:
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv['filters'],
                kernel_size=conv['kernel_size'],
                stride=conv['stride'],
                padding=conv['padding']))
            if not (activation is None):
                layers.append(self._activation_from_name(activation))
            in_channels = conv['filters']

        return nn.Sequential(*layers)
    
    def _build_conv2d(self, input_shape, convs, activation: str = None):
        in_channels = input_shape[0]
        layers = []
        for conv in convs:
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv['filters'],
                kernel_size=conv['kernel_size'],
                stride=conv['stride'],
                padding=conv['padding']))
            if activation is not None:
                layers.append(self._activation_from_name(activation))
            in_channels = conv['filters']

        return nn.Sequential(*layers)

    def _build_mlp(self, input_size: int, units: [int], activation: str = None):
        layers = []
        in_size = input_size
        for unit in units:
            layers.append(nn.Linear(in_features=in_size, out_features=unit))
            if not (activation is None):
                layers.append(self._activation_from_name(activation))
            in_size = unit

        return nn.Sequential(*layers)
