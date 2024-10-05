from random import random

import torchaudio
from torch import Tensor, nn


class TimeMasking(nn.Module):
    def __init__(self, prob: float, time_mask_param: int = 15):
        super().__init__()
        self.prob = prob
        self.masking = torchaudio.transforms.TimeMasking(
            time_mask_param, iid_masks=True
        )

    def __call__(self, data: Tensor):
        return self.masking(data) if self.prob < random() else data
