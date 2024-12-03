import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t


class MinPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self._max_pool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: Tensor):
        return -self._max_pool1d(-x)