import gzip
import os
import urllib.request
import zipfile
from pathlib import Path

import wget

URL_MIT = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
PATH_MIT = "data/mit/reverb/Audio"

URL_NOISE = "https://zenodo.org/records/1227121/files/OOFFICE_16k.zip?download=1"
PATH_NOISE = "data/noise/OOFFICE"

URL_LM = "https://www.openslr.magicdatatech.com/resources/11/4-gram.arpa.gz"
PATH_LM = "data/lm/4-gram.arpa"

URL_VOCAB = "https://www.openslr.magicdatatech.com/resources/11/librispeech-vocab.txt"
PATH_VOCAB = "data/lm/librispeech-vocab.txt"

URL_CORPUS = (
    "https://www.openslr.magicdatatech.com/resources/11/librispeech-lm-norm.txt.gz"
)
PATH_CORPUS = "data/lm/librispeech-lm-norm.txt"


def download_mit() -> str:
    """
    Download reverb audio dataset to impulse response augmentation.

    Returns:
        PATH_NOISE (str): path to dir with wav-files.
    """
    path_dir = Path("data/mit").absolute().resolve()
    path_dir.mkdir(exist_ok=True, parents=True)

    path_dir = path_dir / "reverb"

    if path_dir.exists():
        return path_dir
    path_dir.mkdir(exist_ok=True, parents=True)

    path_zip = path_dir / "Audio.zip"

    wget.download(url=URL_MIT, out=str(path_zip))

    with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
        zip_ref.extractall(str(path_dir))

    os.remove(path_zip)

    return PATH_MIT


def download_noise() -> str:
    """
    Download noise audio dataset.

    Returns:
        PATH_NOISE (str): path to dir with wav-files.
    """
    path_dir = Path("data").absolute().resolve()
    path_dir.mkdir(exist_ok=True, parents=True)

    path_dir = path_dir / "noise"

    if path_dir.exists():
        return path_dir
    path_dir.mkdir(exist_ok=True, parents=True)

    path_zip = path_dir / "OOFFICE_16k.zip"

    wget.download(url=URL_NOISE, out=str(path_zip))

    with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
        zip_ref.extractall(str(path_dir))

    os.remove(path_zip)

    return PATH_NOISE


def download_lm() -> str:
    """
    Download 4-gram lm pre-trained on librispeech corpus.

    Returns:
        PATH_LM (str): path to lm.
    """
    path_gzip = Path("data/lm").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    path_file = path_gzip / "4-gram.arpa"
    if path_file.exists():
        return PATH_LM

    try:
        with urllib.request.urlopen(URL_LM) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()

        with open(path_file, "wb") as f:
            f.write(file_content)

    except Exception as e:
        print(e)

    return PATH_LM


def download_vocab() -> str:
    """
    Download vocab of librispeech.

    Returns:
        PATH_VOCAB (str): path to vocab.
    """
    if Path(PATH_VOCAB).exists():
        return PATH_VOCAB

    urllib.request.urlretrieve(URL_VOCAB, PATH_VOCAB)

    return PATH_VOCAB


def download_lm_corpus() -> str:
    """
    Download text corpus of the 4-gram lm.

    Returns:
        PATH_CORPUS (str): path to text corpus.
    """
    with urllib.request.urlopen(URL_CORPUS) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(PATH_CORPUS, "wb") as f:
        f.write(file_content)

    return PATH_CORPUS
