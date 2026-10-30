# IT Support Ticket Intelligence

This is a machine learning project for classifying IT support
tickets from their descriptions. The final project will use TF-IDF, Random Forest,
scikit-learn, and Streamlit.

## Project status

The data download, inspection, train/test split, and baseline model are complete.
Random Forest training and the Streamlit app are still in progress.

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

## Baseline model

The baseline is a scikit-learn pipeline with TF-IDF and Logistic Regression. Keeping
both steps in one pipeline ensures that the TF-IDF vocabulary is learned from the
training text only.

| Metric | Result |
| --- | ---: |
| Accuracy | 0.7458 |
| Macro precision | 0.7108 |
| Macro recall | 0.7606 |
| Macro F1 | 0.7252 |

Macro metrics give every category equal importance, which is useful here because the
dataset is imbalanced. These results are the reference point for the Random Forest.

## Project structure

```text
data/raw/              Downloaded CSV files (not committed)
scripts/download_data.py
scripts/inspect_data.py
src/data.py            Data loading and train/test split
src/training.py        TF-IDF and Logistic Regression baseline
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

Train and evaluate the current baseline:

```powershell
python -m src.training
```

## Running the app

The Streamlit app has not been added yet. Once the model and app are ready, it will
run with:

```powershell
streamlit run app.py
```

The app will open in a browser, where you can enter a ticket description and receive
a predicted category.
