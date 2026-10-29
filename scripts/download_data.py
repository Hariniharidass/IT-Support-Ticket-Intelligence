"""Download the IT support ticket dataset from Zenodo."""

from pathlib import Path
from urllib.request import urlretrieve

ZENODO_URL = "https://zenodo.org/records/7648117/files"
DATA_DIR = Path("data/raw")
FILES = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        url = f"{ZENODO_URL}/{filename}?download=1"
        destination = DATA_DIR / filename
        print(f"Downloading {filename}...")
        urlretrieve(url, destination)

    print(f"Dataset saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
