import gzip
import urllib.request


def download_file(url):
    # code from https://stackoverflow.com/questions/3548495/download-extract-and-read-a-gzip-file-in-python
    out_file = "librispeech-lm-norm.txt"
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(out_file, "wb") as f:
        f.write(file_content)
        return out_file
