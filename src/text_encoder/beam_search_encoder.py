import re
from abc import abstractmethod
from collections import defaultdict
from string import ascii_lowercase
from typing import List

import kenlm
import numpy as np
from scipy.special import softmax
from torch import Tensor

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class BeamSearchEncoder(CTCTextEncoder):
    def __init__(self, use_lm: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_lm = use_lm

        if use_lm:
            self.lm = kenlm.Model("lm/test.arpa")

    @abstractmethod
    def decode(
        self, logits: Tensor, log_probs_length: Tensor, beam_size: int = 50
    ) -> List[str]:
        """
        Beam search decoding.

        Args:
            logits (Tensor): Tensor of shape (B, M, len(alphabet)) contains logits
            log_probs_length (Tensor):  Tensor of shape (B,) contains length of spectrogram
            beam_size (int): Number of words to save
        Returns:
            result (list[str]): decoded texts.
        """

        logits = logits.cpu().numpy()
        result = [
            self._beam_search(inds[: int(ind_len)], beam_size)
            for inds, ind_len in zip(logits, log_probs_length.numpy())
        ]

        return result

    def _beam_search(self, logits: np.ndarray, beam_size: int = 50) -> str:
        """
        Beam search.

        Args:
            logits (np.ndarray): Matrix of shape (len(alphabet), M) contains logits
            beam_size (int): Number of sentences to save
        Returns:
            text (str): decoded text.
        """

        beam_dict = {("", self.EMPTY_TOK): 1.0}

        for prob in logits:
            beam_dict = self._extend(prob, beam_dict)
            beam_dict = self._cut(beam_dict, beam_size)

        final_dict = defaultdict(float)

        for (position, last_char), prob in beam_dict.items():
            final_dict[
                (position + last_char).replace(self.EMPTY_TOK, "").strip()
            ] += prob

        result = []
        for sentence, prob in final_dict.items():
            result.append([sentence, 10 ** self.lm.score(sentence) * 0.2 + 0.8 * prob])

        return sorted(result, key=lambda x: -x[1])[0][0]

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
