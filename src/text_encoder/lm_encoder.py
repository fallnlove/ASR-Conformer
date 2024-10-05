import multiprocessing
import re
from abc import abstractmethod
from collections import defaultdict
from string import ascii_lowercase
from typing import List

import numpy as np
import torch
from pyctcdecode import build_ctcdecoder
from scipy.special import softmax
from torch import Tensor
from torchaudio.models.decoder import download_pretrained_files

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class LMEncoder(CTCTextEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        files = download_pretrained_files("librispeech-4-gram")

        vocab = self.vocab
        vocab[0] = ""
        for i in range(4, len(vocab)):
            if type(self.vocab[i]) != str or len(self.vocab[i]) == 1:
                vocab[i] = vocab[i].upper()
            else:
                vocab[i] = vocab[i].upper()

        self.decoder = build_ctcdecoder(
            vocab,
            kenlm_model_path=files.lm,
        )

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

        # log_probs = log_probs.transpose(1, 2).cpu().numpy()
        probs = [
            inds[: int(ind_len)].numpy()
            for inds, ind_len in zip(log_probs, log_probs_length.numpy())
        ]
        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = self.decoder.decode_batch(pool, probs)

        return [i.strip().lower() for i in text_list]
