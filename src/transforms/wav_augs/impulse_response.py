from random import random

import torch
import torch_audiomentations
from torch import Tensor, nn

from src.utils.download import download_mit


class ImpulseResponse(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        data_dir = download_mit()
        self._aug = torch_audiomentations.ApplyImpulseResponse(
            data_dir, *args, **kwargs
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
