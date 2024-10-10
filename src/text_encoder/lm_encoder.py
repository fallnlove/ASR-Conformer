import multiprocessing
from abc import abstractmethod
from typing import List, Optional

from pyctcdecode import build_ctcdecoder
from scipy.special import softmax
from torch import Tensor

from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.download import download_lm, download_vocab


class LMEncoder(CTCTextEncoder):
    def __init__(self, beam_size: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size

        path_model = download_lm()
        self.normalize_model_(path_model)

        vocab_path = download_vocab()
        unigram = self.normalize_vocab_(vocab_path)

        vocab = [i.lower() for i in self.vocab]
        vocab[0] = ""

        self.lm = build_ctcdecoder(
            vocab,
            path_model,
            unigram,
        )

    @abstractmethod
    def decode(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        beam_size: Optional[int] = None,
    ) -> List[str]:
        """
        LM based beam search decoding.

        Args:
            log_probs (Tensor): Tensor of shape (B, M, len(alphabet)) contains logits
            log_probs_length (Tensor):  Tensor of shape (B,) contains length of spectrogram
            beam_size (Optional[int]): Number of words to save
        Returns:
            result (list[str]): decoded texts.
        """

        log_probs = [
            log_prob[:length].cpu().numpy()
            for log_prob, length in zip(log_probs, log_probs_length)
        ]

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pred_lm = self.lm.decode_batch(
                pool,
                log_probs,
                beam_width=self.beam_size if beam_size is None else beam_size,
            )
            return [" ".join(sentence.split()) for sentence in pred_lm]

    def normalize_model_(self, path_model):
        with open(path_model) as f_in:
            lm = f_in.read()
        with open(path_model, "w") as f_out:
            f_out.write(lm.lower().replace('"', "").replace("'", ""))

    def normalize_vocab_(self, path_vocab):
        unigram = []
        with open(path_vocab) as f:
            for line in f.readlines():
                line.lower().strip().replace("'", "").replace('"', "").replace("\n", "")
                if line == "":
                    continue
                unigram.append(line)
        return unigram
