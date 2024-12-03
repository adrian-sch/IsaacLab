from .residual_block import ResidualBlock1d
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t, Union


class ImpalaSequential1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1, padding: Union[str, _size_1_t] = 0):
        super().__init__()
        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self._max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self._res_block1 = ResidualBlock1d(out_channels)
        self._res_block2 = ResidualBlock1d(out_channels)

    def forward(self, x: Tensor):
        x = self._conv(x)
        x = self._max_pool(x)
        x = self._res_block1(x)
        x = self._res_block2(x)
        return x