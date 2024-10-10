import torch
from torch import Tensor, nn

from src.model.conformer.basic_blocks import ConformerBlock, SubSampling


class Conformer(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        num_layers: int = 6,
        hidden_dim: int = 144,
        num_heads_transformer: int = 4,
        expansion_factor_ff: int = 4,
        expansion_factor_conv: int = 2,
        kernel_size: int = 31,
        p_drop: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.subsampling = SubSampling(hidden_dim=hidden_dim)

        shape_after_subsamp = ((n_feats - 1) // 2 - 1) // 2 * hidden_dim

        self.in_layer = nn.Sequential(
            nn.Linear(in_features=shape_after_subsamp, out_features=hidden_dim),
            nn.Dropout(p=p_drop),
        )

        self.body = nn.ModuleList(
            [
                ConformerBlock(
                    input_dim=hidden_dim,
                    num_heads=num_heads_transformer,
                    expansion_factor_ff=expansion_factor_ff,
                    expansion_factor_conv=expansion_factor_conv,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    max_len=max_len,
                )
                for _ in range(num_layers)
            ],
        )

        self.fc = nn.Linear(in_features=hidden_dim, out_features=n_tokens)

    def forward(self, spectrogram: Tensor, spectrogram_length, **batch) -> Tensor:
        """
        Args:
            spectrogram (Tensor): Tensor of shape (B, C, T).
            spectrogram_length (Tensor): Lengths tensor of shape (B,).
        Returns:
            (dict): Logits of shape (B, T / 4, hidden_dim).
        """

        log_probs_length = self.transform_input_lengths(spectrogram_length)
        padding_mask = self._create_padding_mask(log_probs_length)
        spectrogram = spectrogram.transpose(1, 2)

        out = self.subsampling(spectrogram)
        out = self.in_layer(out)

        for layer in self.body:
            out = layer(out, padding_mask=padding_mask)

        out = self.fc(out)

        log_probs = nn.functional.log_softmax(out, dim=-1)

        return {
            "log_probs": log_probs,
            "log_probs_length": log_probs_length,
        }

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return ((input_lengths - 1) // 2 - 1) // 2

    def _create_padding_mask(self, spectrogram_length):
        B = spectrogram_length.shape[0]
        T = int(torch.max(spectrogram_length).cpu())

        mask = torch.arange(T).expand(B, T).to(spectrogram_length.device)

        mask = mask >= spectrogram_length.unsqueeze(1)

        return mask
