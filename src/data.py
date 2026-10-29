"""Load the ticket data and create a leakage-safe split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path("data/raw")
RANDOM_SEED = 42
SELECTED_FOLD = 4


def load_data():
    """Load the four CSV files and join ticket text with its category."""
    datasets = []

    for split in ["train", "test"]:
        features = pd.read_csv(DATA_DIR / f"X_{split}.csv")
        labels = pd.read_csv(DATA_DIR / f"y_{split}.csv")
        data = features.merge(labels, on="id", validate="one_to_one")
        datasets.append(data)

    return pd.concat(datasets, ignore_index=True)


def normalize_text(text):
    """Create a consistent key for detecting repeated ticket text."""
    return " ".join(text.lower().split())


def split_data(data):
    """Create an 80/20 split while keeping repeated texts together."""
    groups = data["text"].map(normalize_text)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    splits = list(splitter.split(data["text"], data["category_truth"], groups))
    train_index, test_index = splits[SELECTED_FOLD - 1]

    train_data = data.iloc[train_index].reset_index(drop=True)
    test_data = data.iloc[test_index].reset_index(drop=True)
    return train_data, test_data
