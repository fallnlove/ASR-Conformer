import gzip
import urllib.request
from pathlib import Path


def main():
    path_gzip = Path("data/lm").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    path_file = path_gzip / "4-gram.arpa"

    try:
        with urllib.request.urlopen(
            "https://www.openslr.org/resources/11/4-gram.arpa.gz"
        ) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()

        with open(path_file, "wb") as f:
            f.write(file_content)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
