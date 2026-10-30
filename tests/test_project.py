"""Small tests for the data and training code."""

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.data import normalize_text, split_data
from src.training import build_pipeline


def test_normalize_text():
    assert normalize_text("  RESET   Password  ") == "reset password"


def test_split_keeps_repeated_text_together():
    rows = []
    for number in range(10):
        category = "A" if number % 2 == 0 else "B"
        rows.append({"text": f"Ticket {number}", "category_truth": category})
        rows.append({"text": f"TICKET {number}", "category_truth": category})

    data = pd.DataFrame(rows)
    train_data, test_data = split_data(data)

    train_groups = set(train_data["text"].map(normalize_text))
    test_groups = set(test_data["text"].map(normalize_text))
    assert train_groups.isdisjoint(test_groups)


def test_pipeline_accepts_ticket_text():
    model = build_pipeline(DummyClassifier(strategy="most_frequent"))
    texts = ["password issue", "password problem", "file issue", "file problem"]
    labels = ["Account", "Account", "Files", "Files"]

    model.fit(texts, labels)
    prediction = model.predict(["password issue"])

    assert len(prediction) == 1
