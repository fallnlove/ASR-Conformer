import gdown

URLS = {
    "https://drive.google.com/uc?id=1rTln0R_vpfWHNpaykGsXix5aqMNrGcmN": "data/models/best_ctc_conformer.pth",
    "https://drive.google.com/uc?id=1xJavDOhOgpXw2rIDZiRTVuS8bqaK1zgH": "data/models/conformer_before_fine_tuning.pth",
}


def main():
    for url, path in URLS.keys():
        gdown.download(url, path)


if __name__ == "__main__":
    main()
