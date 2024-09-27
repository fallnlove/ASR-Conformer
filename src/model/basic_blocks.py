from typing import Optional

import torch
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(
        self, input_dim: int = 144, expansion_factor: int = 4, p_drop: float = 0.1
    ):
        super().__init__()

        hidden_dim = input_dim * expansion_factor

        self.ffn = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_dim),
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=p_drop),
            nn.Linear(in_features=hidden_dim, out_features=input_dim),
            nn.Dropout(p=p_drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """
        return self.ffn(x) / 2 + x


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim: int = 144, max_len: int = 5000):
        super().__init__()

        idx = 1.0 / 10000 ** (torch.arange(0, input_dim, 2) / input_dim)
        pos = torch.arange(0, max_len).reshape(max_len, 1)

        self.embedding = torch.zeros((max_len, input_dim))
        self.embedding[:, 0::2] = torch.sin(pos * idx)
        self.embedding[:, 1::2] = torch.cos(pos * idx)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """
        return x + self.embedding[: x.shape[1], :].to(x.device)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 144,
        num_heads: int = 4,
        p_drop: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim)
        self.pos_encoding = PositionalEncoding(input_dim, max_len=max_len)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
            padding_mask (Optional[Tensor]): padding mask to use in mhsa
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """
        out = self.layer_norm(x)
        out = self.pos_encoding(out)

        out, _ = self.mhsa(
            query=out,
            key=out,
            value=out,
            need_weights=False,
            key_padding_mask=padding_mask,
        )
        out = self.dropout(out)

        return out + x


class Convolution(nn.Module):
    def __init__(
        self,
        input_dim: int = 144,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        p_drop: float = 0.1,
    ):
        super().__init__()

        hidden_dim = input_dim * expansion_factor

        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(num_features=input_dim),
            nn.SiLU(),
            nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1),
            nn.Dropout(p=p_drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """

        out = self.layer_norm(x)
        out = out.transpose(1, 2)  # (B, C, T)
        out = self.conv(out)

        return out.transpose(1, 2) + x  # (B, T, C)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 144,
        num_heads: int = 4,
        expansion_factor_ff: int = 4,
        expansion_factor_conv: int = 2,
        kernel_size: int = 31,
        p_drop: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.ff = FeedForward(
            input_dim=input_dim, expansion_factor=expansion_factor_ff, p_drop=p_drop
        )
        self.mhsa = MultiHeadedSelfAttention(
            input_dim=input_dim, num_heads=num_heads, p_drop=p_drop, max_len=max_len
        )
        self.conv = nn.Sequential(
            Convolution(
                input_dim=input_dim,
                expansion_factor=expansion_factor_conv,
                kernel_size=kernel_size,
                p_drop=p_drop,
            ),
            FeedForward(
                input_dim=input_dim, expansion_factor=expansion_factor_ff, p_drop=p_drop
            ),
            nn.LayerNorm(normalized_shape=input_dim),
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (B, T, C)
        Returns:
            x (Tensor): Tensor of shape (B, T, C)
        """

        out = self.ff(x)
        out = self.mhsa(out, padding_mask=padding_mask.to(x.device))
        out = self.conv(out)

        return out
