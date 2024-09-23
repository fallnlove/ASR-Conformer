import re
from abc import abstractmethod
from string import ascii_lowercase
from typing import List

import torch
from torch import Tensor


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, use_bpe: bool = False, alphabet=None, **kwargs):
        """
        Args:
            use_bpe (bool): flag to use BPE
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.use_bpe = use_bpe
        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    @abstractmethod
    def decode(self, log_probs: Tensor, log_probs_length: Tensor) -> List[str]:
        """
        Decoding wit CTC.

        Args:
            log_probs (Tensor): Tensor of shape (B, len(alphabet), M) contains logits
            log_probs_length (Tensor):  Tensor of shape (B,) contains length of spectrogram
        Returns:
            result (list[str]): decoded texts.
        """

        argmax_inds = log_probs.cpu().argmax(-2).numpy()
        result = [
            self._ctc_decode(inds[: int(ind_len)])
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]

        return result

    def _ctc_decode(self, inds) -> str:
        """
        Decoding wit CTC.

        Args:
            inds (list): list of tokens.
        Returns:
            text (str): decoded text.
        """

        decoded_chars = []
        last_ind = self.char2ind[self.EMPTY_TOK]

        for ind in inds:
            if ind == last_ind:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                decoded_chars.append(self.ind2char[ind])
            last_ind = ind

        return "".join(decoded_chars).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
