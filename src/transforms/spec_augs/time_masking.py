import torchaudio
from torch import Tensor, nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param: int = 100):
        super().__init__()
        self.masking = torchaudio.transforms.TimeMasking(
            time_mask_param, iid_masks=True
        )

    def __call__(self, data: Tensor):
        return self.masking(data)
