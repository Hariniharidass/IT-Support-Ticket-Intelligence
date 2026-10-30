# IT Support Ticket Intelligence

This is a machine learning project for classifying IT support
tickets from their descriptions. The final project will use TF-IDF, Random Forest,
scikit-learn, and Streamlit.

## Project status

The data preparation, model comparison, and model export are complete. The Streamlit
app is still in progress.

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
data/raw/              Downloaded CSV files (not committed)
scripts/download_data.py
scripts/inspect_data.py
src/data.py            Data loading and train/test split
src/training.py        TF-IDF model training and comparison
models/                Generated model file (not committed)
reports/               Generated confusion matrix (not committed)
tests/
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

The Streamlit app has not been added yet. Once the model and app are ready, it will
run with:

```powershell
streamlit run app.py
```

The app will open in a browser, where you can enter a ticket description and receive
a predicted category.
