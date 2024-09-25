from typing import Union

import torch


def collate_fn(dataset_items: list[dict]) -> dict[Union[torch.Tensor, list]]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, list]]): dict, containing batch-version
            of the tensors.
    """

    return {
        "audio": [item["audio"] for item in dataset_items],
        "spectrogram": torch.nn.utils.rnn.pad_sequence(
            [
                item["spectrogram"].squeeze(0).transpose(-1, -2)
                for item in dataset_items
            ],
            batch_first=True,
        ).transpose(-1, -2),
        "spectrogram_length": torch.IntTensor(
            [int(item["spectrogram"].shape[-1]) for item in dataset_items],
        ),
        "text": [item["text"] for item in dataset_items],
        "text_encoded": torch.nn.utils.rnn.pad_sequence(
            [item["text_encoded"].transpose(-1, -2) for item in dataset_items],
            batch_first=True,
        ).squeeze(-1),
        "text_encoded_length": torch.IntTensor(
            [int(item["text_encoded"].shape[-1]) for item in dataset_items],
        ),
        "audio_path": [item["audio_path"] for item in dataset_items],
        "sample_rate": [item["sample_rate"] for item in dataset_items],
    }
