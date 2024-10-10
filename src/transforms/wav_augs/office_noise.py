import os
from random import choice, randint, random

import torch
import torchaudio
from torch import Tensor, nn

from src.utils.download import download_noise


class OfficeNoise(nn.Module):
    def __init__(self, p: float, snr: float = 15):
        super().__init__()
        self.p = p
        self.snr = Tensor([snr])
        self.data_dir = download_noise()

    def __call__(self, data: Tensor):
        file = choice(os.listdir(self.data_dir))
        noise, _ = torchaudio.load(self.data_dir + file)

        left = randint(0, noise.shape[-1] - data.shape[-1] - 1)
        noise = noise[..., left : left + data.shape[-1]]

        return (
            torchaudio.functional.add_noise(data, noise, self.snr)
            if self.p < random()
            else data
        )
