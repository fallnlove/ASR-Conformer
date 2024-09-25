import re
from abc import abstractmethod
from collections import defaultdict
from string import ascii_lowercase
from typing import List

import numpy as np
import torch
from scipy.special import softmax
from torch import Tensor

from src.text_encoder import CTCTextEncoder


class BeamSearchEncoder(CTCTextEncoder):
    @abstractmethod
    def decode(
        self, log_probs: Tensor, log_probs_length: Tensor, beam_size: int = 50
    ) -> List[str]:
        """
        Beam search decoding.

        Args:
            log_probs (Tensor): Tensor of shape (B, M, len(alphabet)) contains logits
            log_probs_length (Tensor):  Tensor of shape (B,) contains length of spectrogram
            beam_size (int): Number of words to save
        Returns:
            result (list[str]): decoded texts.
        """

        log_probs = log_probs.transpose(1, 2).cpu().numpy()
        result = [
            self._beam_search(inds, beam_size)[: int(ind_len)]
            for inds, ind_len in zip(log_probs, log_probs_length.numpy())
        ]

        return result

    def _beam_search(self, log_probs: np.ndarray, beam_size: int = 50) -> str:
        """
        Beam search.

        Args:
            log_probs (np.ndarray): Matrix of shape (len(alphabet), M) contains logits
            beam_size (int): Number of sentences to save
        Returns:
            text (str): decoded text.
        """

        probs = softmax(log_probs)
        beam_dict = {("", self.EMPTY_TOK): 1.0}

        for prob in probs.T:
            beam_dict = self._extend(prob, beam_dict)
            beam_dict = self._cut(beam_dict, beam_size)

        final_dict = defaultdict(float)

        for (position, last_char), prob in beam_dict.items():
            final_dict[
                (position + last_char).replace(self.EMPTY_TOK, "").strip()
            ] += prob

        return sorted(final_dict.items(), key=lambda x: -x[1])[0][0]

    def _extend(self, probs: np.ndarray, beam_dict: dict) -> dict:
        """
        Extend part of beam search.

        Args:
            probs (np.ndarray): Vector of shape (len(alphabet),) contains probabilities
            beam_dict (dict): Dict contains pairs (position, last_char) with probabilities
        Returns:
            new_dict (dict): Dict on the next step
        """
        new_dict = defaultdict(float)

        for (position, last_char), prob in beam_dict.items():
            for idx, p in enumerate(probs):
                new_char = self.ind2char[idx]
                if new_char == last_char:
                    new_dict[(position, last_char)] += prob * p
                else:
                    new_dict[
                        ((position + last_char).replace(self.EMPTY_TOK, ""), new_char)
                    ] += (prob * p)

        return new_dict

    def _cut(self, beam_dict: dict, beam_size: int) -> dict:
        """
        Cut dict of beam search.

        Args:
            beam_dict (dict): Dict contains pairs (position, last_char) with probabilities
            beam_size (int): Number of sentences to save
        Returns:
            cut_dict (dict): Cut dict
        """

        return dict(sorted(beam_dict.items(), key=lambda x: -x[1])[:beam_size])
