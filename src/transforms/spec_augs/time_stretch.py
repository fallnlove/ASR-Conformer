from random import random

import torchaudio
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, rate_std: float = 0.1):
        super().__init__()
        self.rate_std = rate_std
        self.stretch = torchaudio.transforms.TimeStretch()

    def __call__(self, data: Tensor):
        return self.stretch(data, overriding_rate=self._rand())

    def _rand(self):
        min = 1.0 - self.rate_std
        max = 1.0 + self.rate_std

        return (random() * (max - min)) + min
