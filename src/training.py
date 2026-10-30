"""Train and evaluate a simple ticket classification baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from src.data import RANDOM_SEED, load_data, split_data


def build_baseline():
    """Create a TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=15_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_model(y_true, y_pred):
    """Calculate metrics that account for the imbalanced categories."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():
    data = load_data()
    train_data, test_data = split_data(data)

    model = build_baseline()

    # fit() learns the TF-IDF vocabulary from training text only.
    model.fit(train_data["text"], train_data["category_truth"])
    predictions = model.predict(test_data["text"])
    metrics = evaluate_model(test_data["category_truth"], predictions)

    print("Logistic Regression baseline")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
