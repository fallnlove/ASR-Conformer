# Automatic Speech Recognition (ASR) with Conformer

Implementation of the ASR model based on the article [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/pdf/2005.08100).

<img src="https://user-images.githubusercontent.com/1694368/101230439-4699f080-36e0-11eb-839f-5d70d21f1310.png" width="400"/>

||WER||CER||
|-|-|-|-|-|
|Inference type|test-clean|test-other|test-clean|test-other|
|CTCArgmax|19.37|36.06|6.80|15.59|
|Beam Search(beam_size=16)|19.04|35.43|6.65|15.23|
|LM(beam_size=10000)|12.84|27.02|5.34|14.09|

## Installation

1. Install dependencies

```bash
pip install -r ./requirements.txt
```

2. Install checkpoints and pre-trained model

```bash
python3 scripts/download_models.py
```

## Training

If you want to reproduce training process following steps below.

1. Train bpe encoder(default NUM_TOKENS=128)

```bash
pyhton3 src/bpe/bpe_train.py -cn=baseline num_tokens=NUM_TOKENS
```

2. Train Conformer on mix of Librispeech train-clean-100 and train-clean-360

```bash
pyhton3 train.py -cn=conformer_lib_sp
```

3. Fine-tune model on mix of Librispeech train-clean and train-other with [SpecAugment](https://arxiv.org/pdf/1904.08779)

```bash
pyhton3 train.py -cn=conformer_fine_tune trainer.from_pretrained=PATH_TO_LAST_CHECKPOINT
```

Note: default `PATH_TO_LAST_CHECKPOINT` is path to downloaded checkpoint, but you can use checkpoint of your training at the previous step.

## Inference

Repository provide various version of inference on Librispeech test set.

1. Inference with Argmax

```bash
pyhton3 inference.py -cn=inference
```

2. Inference with BeamSearch

```bash
pyhton3 inference.py -cn=inference_beam_search
```

3. Inference with pre-trained LM. I use pre-trained LM on Librispeech train text corpus and [external library](https://github.com/kensho-technologies/pyctcdecode) for LM-based beam search

```bash
pyhton3 inference.py -cn=inference_lm
```

## Custom datatset inference

If you want to inference model on custom dataset run following commands.

```bash
python3 inference.py -cn=custom_inference \
datasets.dataset_dir=PATH_TO_CUSTOM_DATASET \
inferencer.save_path=PATH_TO_DIR_TO_SAVE_PREDICTIONS
```

Note: `datasets.dataset_dir` should contain folder `audio` and may contain folder `transcriptions` for ground truth texts.

To calculate metrics(for ex. CER and WER) run this.

```bash
python3 scripts/calculate_wer_cer.py -cn=calc_wer_cer \
prediction_path=PATH_TO_DIR_WITH_PREDICTIONS \
target_path=PATH_TO_GROUND_TRUTH_TEXTS
```

## Unit Tests

There are unit tests for functions `collate_fn`, `calc_wer`, `calc_cer` and for hand-crafted
beam search. You can run tests with the following command.

```bash
pytest test.py
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
