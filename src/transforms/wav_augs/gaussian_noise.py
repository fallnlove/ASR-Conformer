import torchaudio
from torch import Tensor, distributions, nn


class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.05, snr: float = 10):
        super().__init__()
        self.noise = distributions.Normal(0, std)
        self.snr = Tensor([snr])

    def __call__(self, data: Tensor):
        return torchaudio.functional.add_noise(
            data, self.noise.sample(data.size), self.snr
        )
