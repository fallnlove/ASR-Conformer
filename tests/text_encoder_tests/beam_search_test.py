import numpy as np
import torch

from src.text_encoder import BeamSearchEncoder, CTCTextEncoder


def test_ctc_decode():
    encoder = BeamSearchEncoder()
    base_encoder = CTCTextEncoder()

    assert (
        encoder._beam_search(
            np.array(
                [
                    [0.5, 0.4, 0.1, 0.2],
                    [0.2, 0.1, 0.3, 0.7],
                    [0.3, 0.5, 0.6, 0.1],
                ]
            ),
            1,
        )
        == base_encoder.decode(
            torch.Tensor(
                [
                    [
                        [0.5, 0.4, 0.1, 0.2],
                        [0.2, 0.1, 0.3, 0.7],
                        [0.3, 0.5, 0.6, 0.1],
                    ]
                ]
            ),
            torch.Tensor([4]),
        )[0]
    )

    assert (
        encoder._beam_search(
            np.array(
                [
                    [0.5, 0.01, 0.5],
                    [0.49, 0.49, 0.49],
                    [0.01, 0.5, 0.01],
                ]
            ),
            10,
        )
        == "a"
    )
