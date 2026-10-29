"""Inspect the dataset before writing preprocessing code."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")


def main():
    all_splits = {}

    for split in ["train", "test"]:
        features = pd.read_csv(DATA_DIR / f"X_{split}.csv")
        labels = pd.read_csv(DATA_DIR / f"y_{split}.csv")
        data = features.merge(labels, on="id", validate="one_to_one")
        all_splits[split] = data

        print(f"\n{split.upper()} SET")
        print(f"Feature shape: {features.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Feature columns: {features.columns.tolist()}")
        print(f"Label columns: {labels.columns.tolist()}")
        print(f"Missing values:\n{data.isna().sum()}")
        print(f"Duplicate ticket texts: {data['text'].duplicated().sum()}")
        print(f"Category counts:\n{data['category_truth'].value_counts()}")

    repeated_across_splits = set(all_splits["train"]["text"]) & set(
        all_splits["test"]["text"]
    )
    print(f"\nTicket texts found in both splits: {len(repeated_across_splits)}")


if __name__ == "__main__":
    main()
