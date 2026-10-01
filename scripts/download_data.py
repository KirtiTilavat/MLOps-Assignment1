"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/.

Usage:
    python scripts/download_data.py

Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_FILE = RAW_DIR / "processed.cleveland.data"


def download(url: str = URL, dest: Path = RAW_FILE, force: bool = False) -> Path:
    """Download the raw dataset file. Skips the download if it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Already present: {dest}")
        return dest

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    n_lines = sum(1 for _ in dest.open())
    print(f"Saved {n_lines} rows to {dest}")
    return dest


if __name__ == "__main__":
    download(force="--force" in sys.argv)
