import re
from abc import abstractmethod
from collections import defaultdict
from string import ascii_lowercase
from typing import List, Optional

import kenlm
import numpy as np
from torch import Tensor
from torch.nn.functional import softmax

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class BeamSearchEncoder(CTCTextEncoder):
    def __init__(self, use_lm: bool = False, beam_size: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_lm = use_lm
        self.beam_size = beam_size

        if use_lm:
            self.lm = kenlm.Model("data/lm/4-gram.arpa")

    @abstractmethod
    def decode(
        self, logits: Tensor, log_probs_length: Tensor, beam_size: Optional[int] = None
    ) -> List[str]:
        """
        Beam search decoding.

        Args:
            logits (Tensor): Tensor of shape (B, M, len(alphabet)) contains logits
            log_probs_length (Tensor):  Tensor of shape (B,) contains length of spectrogram
            beam_size (Optional[int]): Number of words to save
        Returns:
            result (list[str]): decoded texts.
        """

        logits = softmax(logits, -1).cpu().numpy()
        result = [
            self._beam_search(
                inds[: int(ind_len)], self.beam_size if beam_size is None else beam_size
            )
            for inds, ind_len in zip(logits, log_probs_length.numpy())
        ]

        return result

    def _beam_search(self, logits: np.ndarray, beam_size) -> str:
        """
        Beam search.

        Args:
            logits (np.ndarray): Matrix of shape (M, len(alphabet)) contains logits
            beam_size (int): Number of sentences to save
        Returns:
            text (str): decoded text.
        """

        beam_dict = {("", self.EMPTY_TOK): 1.0}

        for prob in logits:
            beam_dict = self._extend(prob, beam_dict)
            beam_dict = self._cut(beam_dict, beam_size)

        final_dict = defaultdict(float)

        for (position, _), prob in beam_dict.items():
            final_dict[position] = prob

        result = []
        for sentence, prob in final_dict.items():
            result.append(
                [
                    sentence,
                    self.lm.score(sentence) / 2 + prob if self.use_lm else prob,
                ]
            )

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

        for idx, p in enumerate(probs):
            cur_char = self.ind2char[idx]
            for (prefix, last_char), prob in beam_dict.items():
                if cur_char == last_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix

                new_dict[(new_prefix, cur_char)] += prob * p

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
