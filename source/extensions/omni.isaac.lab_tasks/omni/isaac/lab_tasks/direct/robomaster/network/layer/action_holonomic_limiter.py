import torch.nn as nn
from torch import Tensor
import torch



class ActionHolonomicLimiter(nn.Module):
    def __init__(self, begin: float = 0.0, end: float = 1.0):
        super().__init__()

        if begin > end:
            raise ValueError("begin {} must be smaller then end {}".format(begin, end))

        area = end - begin
        self._shift = end - area * 0.5
        self._scale = 10.0 / area
        self._sig = nn.Sigmoid()

    def forward(self, x: Tensor):
        if x.shape[1] != 3:
            raise ValueError(f"Expected input tensor with shape [*, 3], but got shape {x.shape}")

        # Create a copy of x to avoid in-place operations
        out = x.clone()
        out[:, 1] = x[:, 1] * (1.0 - self._sig(self._scale * (x[:, 0] - self._shift)))
        return out