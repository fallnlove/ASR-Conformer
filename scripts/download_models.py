from pathlib import Path

import gdown

URLS = {
    "https://drive.google.com/uc?id=1rTln0R_vpfWHNpaykGsXix5aqMNrGcmN": "data/models/best_ctc_conformer.pth",
    "https://drive.google.com/uc?id=1xJavDOhOgpXw2rIDZiRTVuS8bqaK1zgH": "data/models/conformer_before_fine_tuning.pth",
}


def main():
    path_gzip = Path("data/models").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)


if __name__ == "__main__":
    main()
