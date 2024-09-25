import torchaudio
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param: int = 27):
        super().__init__()
        self.masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param, iid_masks=True
        )

    def __call__(self, data: Tensor):
        return self.masking(data)
