import os
import zipfile
from pathlib import Path

import wget

URL_MIT = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
URL_NOISE = "https://zenodo.org/records/1227121/files/OOFFICE_16k.zip?download=1"


def download_mit() -> str:
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

    return "data/mit/reverb/Audio"


def download_noise() -> str:
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

    return "data/noise/OOFFICE"
