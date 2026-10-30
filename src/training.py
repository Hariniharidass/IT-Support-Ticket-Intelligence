"""Train and evaluate a simple ticket classification baseline."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from src.data import RANDOM_SEED, load_data, split_data

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def build_pipeline(classifier):
    """Connect TF-IDF preprocessing to a classifier."""
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
            ("classifier", classifier),
        ]
    )


def build_models():
    """Create the baseline and Random Forest pipelines."""
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(
                max_iter=1_000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
            )
        ),
    }


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


def save_random_forest_outputs(model, y_true, y_pred):
    """Save the fitted pipeline and its confusion matrix."""
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_DIR / "ticket_classifier.joblib")

    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        xticks_rotation=45,
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Random Forest confusion matrix")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def main():
    data = load_data()
    train_data, test_data = split_data(data)

    for model_name, model in build_models().items():
        # fit() learns the TF-IDF vocabulary from training text only.
        model.fit(train_data["text"], train_data["category_truth"])
        predictions = model.predict(test_data["text"])
        metrics = evaluate_model(test_data["category_truth"], predictions)

        print(f"\n{model_name}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        if model_name == "Random Forest":
            save_random_forest_outputs(
                model,
                test_data["category_truth"],
                predictions,
            )
            print("Saved Random Forest model and confusion matrix")


if __name__ == "__main__":
    main()
