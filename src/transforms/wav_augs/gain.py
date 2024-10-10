from random import random

import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, p: float, *args, **kwargs):
        super().__init__()
        self.p = p
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1) if self.p < random() else data
