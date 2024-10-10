from random import random

import torchaudio
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, p: float, rate_std: float = 0.1, *args, **kwargs):
        super().__init__()
        self.p = p
        self.rate_std = rate_std
        self.stretch = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        return (
            self.stretch(data.unsqueeze(0), overriding_rate=self._rand())
            .squeeze(0)
            .absolute()
            if self.p < random()
            else data
        )

    def _rand(self):
        min = 1.0 - self.rate_std
        max = 1.0 + self.rate_std

        return (random() * (max - min)) + min
