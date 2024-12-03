import torch.nn as nn
from torch import Tensor


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self._conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self._conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self._activation1 = nn.ReLU()
        self._activation2 = nn.ReLU()

    def forward(self, x: Tensor):
        residual = x
        x = self._activation1(x)
        x = self._conv1(x)
        x = self._activation2(x)
        x = self._conv2(x)
        return x + residual