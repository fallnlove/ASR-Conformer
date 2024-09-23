import torch

from src.datasets.collate import collate_fn


def collate_test():
    input = [
        {
            "audio": torch.tensor([-1, -0.5, 1, 0, 1]),
            "spectrogram": torch.tensor(
                [
                    [8, 4, 5, 6, 7],
                    [12.3, 13, 4, 1.2, 5],
                ]
            ),
            "text": "abobus",
            "text_encoded": torch.tensor([0, 8, 4, 2, 5]),
            "audio_path": "/home/audio/1",
        },
        {
            "audio": torch.tensor([0, 0.3, -0.2, 1]),
            "spectrogram": torch.tensor(
                [
                    [4, 5, 6],
                    [-12, 1.1, 5.6],
                ]
            ),
            "text": "sound",
            "text_encoded": torch.tensor([1, 3, 5]),
            "audio_path": "/home/audio/2",
        },
    ]

    output = {
        "audio": [
            torch.tensor([-1, -0.5, 1, 0, 1]),
            torch.tensor([0, 0.3, -0.2, 1]),
        ],
        "spectrogram": torch.tensor(
            [
                [
                    [8, 4, 5, 6, 7],
                    [12.3, 13, 4, 1.2, 5],
                ],
                [
                    [4, 5, 6, 0, 0],
                    [-12, 1.1, 5.6, 0, 0],
                ],
            ]
        ),
        "spectrogram_length": torch.Tensor(
            [5, 3],
        ),
        "text": [
            "abobus",
            "sound",
        ],
        "text_encoded": torch.tensor(
            [
                [0, 8, 4, 2, 5],
                [1, 3, 5, 0, 0],
            ]
        ),
        "audio_path": [
            "/home/audio/1",
            "/home/audio/2",
        ],
    }

    assert collate_fn(input) == output
