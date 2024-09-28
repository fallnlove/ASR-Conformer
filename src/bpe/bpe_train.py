import os
import warnings

import hydra
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from utils import download_file

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="../configs", config_name="bpe")
def main(config):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["^", " ", "[UNK]"], vocab_size=config.num_tokens
    )
    tokenizer.pre_tokenizer = Whitespace()

    path = download_file(
        "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
    )

    tokenizer.train([path], trainer)
    tokenizer.save("src/bpe/tokenizer.json")
    os.remove(path)


if __name__ == "__main__":
    main()
