# IT Support Ticket Intelligence

This is a machine learning project for classifying IT support
tickets from their descriptions. The final project will use TF-IDF, Random Forest,
scikit-learn, and Streamlit.

## Project status

The working MVP is complete: it prepares the data, compares two models, saves the
Random Forest pipeline, and provides a Streamlit interface for predictions.

## Dataset

The data comes from Leonardo Santiago Benitez Pereira's
[Classification of IT Support Tickets](https://doi.org/10.5281/zenodo.7648117)
dataset. It contains 2,229 anonymized tickets in seven categories.

The feature files contain `id` and `text`. The label files contain `id` and
`category_truth`.

The Zenodo record does not state a dataset license, so the CSV files are not included
in this repository. Please review the record before downloading or using the data.
The MIT license in this repository applies only to the project code.

## What I found during inspection

- There are no missing ticket descriptions or category labels.
- The categories are imbalanced.
- The original train and test files share 33 identical ticket descriptions.
- The EOL category has 103 rows but only two unique descriptions.

Using the original split could make the results look better because the model could
see the same description during both training and testing. To avoid that, the files
are combined and split again with `StratifiedGroupKFold`. Duplicate descriptions are
kept on the same side of the split.

The current split has 1,816 training rows and 413 test rows, with no repeated text
across the two sets. Only two EOL rows are in the test set, so results for that
category will need to be interpreted carefully.

## Model results

Both models use the same TF-IDF settings and train/test split. Keeping TF-IDF and the
classifier in one pipeline ensures that the vocabulary is learned from training text
only.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7458 | 0.7108 | **0.7606** | **0.7252** |
| Random Forest | 0.7458 | **0.7451** | 0.6732 | 0.7009 |

Random Forest has better macro precision, but Logistic Regression has better macro
recall and macro F1. This shows why a more complex model is not automatically better.
The project will still save Random Forest for the app because it is the requested
model, while documenting the comparison honestly.

## Project structure

```text
app.py                   Streamlit prediction interface
data/raw/              Downloaded CSV files (not committed)
scripts/download_data.py
scripts/inspect_data.py
src/data.py            Data loading and train/test split
src/training.py        TF-IDF model training and comparison
models/                Generated model file (not committed)
reports/               Generated confusion matrix (not committed)
tests/test_project.py  Tests for splitting and the text pipeline
requirements.txt
requirements-dev.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Download and inspect the dataset:

```powershell
python scripts/download_data.py
python scripts/inspect_data.py
```

Train and evaluate both models:

```powershell
python -m src.training
```

This command also creates:

- `models/ticket_classifier.joblib`, containing the fitted TF-IDF and Random Forest
  pipeline
- `reports/confusion_matrix.png`, showing the Random Forest predictions by category

Both files are generated locally and ignored by Git. Keeping TF-IDF inside the saved
pipeline ensures that the app uses the same vocabulary as the trained model.

## Running the app

Train the model first if `models/ticket_classifier.joblib` does not exist:

```powershell
python -m src.training
```

Then start the app:

```powershell
streamlit run app.py
```

Open the local address shown by Streamlit, usually `http://localhost:8501`. Enter a
ticket description and select **Predict category** to see the model's result.

## Tests

Run the automated tests with:

```powershell
python -m pytest tests -q
```

The tests check text normalization, duplicate-safe splitting, and whether the TF-IDF
pipeline can accept ticket text and return a prediction.

## Limitations

- The dataset is small, imbalanced, and contains many repeated descriptions.
- EOL has only two unique descriptions, so its evaluation is not reliable.
- The tickets come from one support environment and may not represent other companies.
- The data is multilingual, but the models do not include language-specific handling.
- Random Forest probabilities are not calibrated.
- This project is for learning and should not replace human ticket review.
