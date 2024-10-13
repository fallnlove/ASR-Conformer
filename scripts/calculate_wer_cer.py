import os
import warnings
from pathlib import Path

import hydra
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from utils import calc_cer, calc_wer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None, config_path="../src/configs/scripts", config_name="calc_wer_cer"
)
def main(config):
    pred_dir = Path(config.prediction_path)
    targer_dir = Path(config.target_path)

    cer = 0
    wer = 0
    total = 0

    for file in pred_dir.iterdir():
        if file.suffix != ".txt":
            continue
        total += 1
        file_name = file.name

        pred_file = pred_dir / file_name
        with pred_file.open() as f:
            pred = f.read().strip().lower().replace("\n", "")

        target_file = targer_dir / file_name
        with target_file.open() as f:
            target = f.read().strip().lower().replace("\n", "")

        cer += calc_cer(target, pred)
        wer += calc_wer(target, pred)

    print(f"CER: {cer/total}\nWER: {wer/total}")


if __name__ == "__main__":
    main()
