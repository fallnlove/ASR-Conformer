from random import random

import torch
import torchaudio
from torch import Tensor, distributions, nn


class GaussianNoise(nn.Module):
    def __init__(self, prob: float, std: float = 0.05, snr: float = 15):
        super().__init__()
        self.prob = prob
        self.std = std
        self.snr = Tensor([snr])

    def __call__(self, data: Tensor):
        noise = torch.empty_like(data)
        noise.normal_(0, self.std)
        return (
            torchaudio.functional.add_noise(data, noise, self.snr)
            if self.prob < random()
            else data
        )
