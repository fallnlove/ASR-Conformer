from random import random

import torch
import torchaudio
from torch import Tensor, nn


class GaussianNoise(nn.Module):
    def __init__(self, p: float, std: float = 0.05, snr: float = 15):
        super().__init__()
        self.p = p
        self.std = std
        self.snr = Tensor([snr])

    def __call__(self, data: Tensor):
        noise = torch.empty_like(data)
        noise.normal_(0, self.std)
        return (
            torchaudio.functional.add_noise(data, noise, self.snr)
            if random() < self.p
            else data
        )
