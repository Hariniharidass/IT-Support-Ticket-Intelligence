# IT Support Ticket Intelligence

This project classifies IT support tickets from their descriptions. It was built as
a small, beginner-friendly example of text classification using TF-IDF, scikit-learn,
Random Forest, and Streamlit.

## Project status

The MVP downloads and checks the data, compares two models, saves the fitted Random
Forest pipeline, and provides a Streamlit page for trying predictions.

## Dataset

The data comes from Leonardo Santiago Benitez Pereira's
[Classification of IT Support Tickets](https://doi.org/10.5281/zenodo.7648117)
dataset. It contains 2,229 anonymized tickets in seven categories.

The feature files contain `id` and `text`. The label files contain `id` and
`category_truth`.

The Zenodo record does not state a dataset license, so the CSV files are not included
in this repository. Please review the record before downloading or using the data.
The MIT license in this repository applies only to the project code.

## Data inspection and splitting

- There are no missing ticket descriptions or category labels.
- The categories are imbalanced.
- The original train and test files share 33 identical ticket descriptions.
- The EOL category has 103 rows but only two unique descriptions.

The original split is not suitable for a fair evaluation because the model could see
the same description during both training and testing. The four files are therefore
combined and split again with `StratifiedGroupKFold`. Duplicate descriptions are kept
on the same side of the split.

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

Random Forest has better macro precision, while Logistic Regression has better macro
recall and macro F1. The app uses Random Forest because that is the model this project
set out to explore, but the baseline performed better on macro F1.

## Project structure

```text
app.py                    Streamlit prediction interface
data/raw/                 Downloaded CSV files (not committed)
scripts/download_data.py  Dataset download script
scripts/inspect_data.py   Dataset inspection script
src/data.py               Data loading and train/test split
src/training.py           Model training, evaluation, and export
models/                   Generated model file (not committed)
reports/                  Generated confusion matrix (not committed)
tests/test_project.py     Tests for splitting and the text pipeline
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

Both files are generated locally and ignored by Git. TF-IDF is saved inside the
pipeline, so the app always uses the same vocabulary as the trained model.

## Running the app

Train the model first if `models/ticket_classifier.joblib` does not exist:

```powershell
python -m src.training
```

Then start the app:

```powershell
streamlit run app.py
```

Open the address shown by Streamlit, usually `http://localhost:8501`. The page lists
the seven available categories. Enter a ticket description and select **Predict
category** to see the prediction and confidence. Results below 50% confidence are
marked for manual review.

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
- The displayed confidence is useful as a warning signal, not as a guarantee that a
  prediction is correct.
- This project is for learning and should not replace human ticket review.
