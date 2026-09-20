## Setup

Install Poetry 2.1+ and Python 3.13.15.

To install PyEnv and Poetry, if necessary, please follow the official documentation:

* [Poetry](https://python-poetry.org/docs/).
* [PyEnv](https://github.com/pyenv/pyenv#installation).

```sh
pyenv install -s 3.13.15
pyenv global 3.13.15 # or local, if you do not want to mess with your environment. 
poetry env use "$(pyenv which python)"
poetry install
```

Run these commands from the project directory; `.python-version` selects the
project interpreter. Poetry creates the environment in `.venv`. No global
Python selection is required. Commit `poetry.lock` with the project.

## Commands

```sh
poetry run dialog-acts --help
poetry run dialog-acts train --help
poetry run dialog-acts eval --help
poetry run pytest
```

# Training the model

```sh
poetry run dialog-acts train ./data/dialog_acts.dat
```

# Rule-based baseline

Evaluate the hand-written keyword baseline on both the original (random) and
the grouped (no duplicate utterances across train/test) 85/15 splits. Both
splits are stratified on the dialog act. `--errors N` also lists the N most
frequent misclassified test utterances.

```sh
poetry run dialog-acts baseline
poetry run dialog-acts baseline --split grouped --errors 20
```

# Classifying your own utterances

```sh
poetry run dialog-acts predict
```

Type an utterance and press enter to see the predicted dialog act; an empty
line or `exit` quits.

# Splits and metrics for other classifiers

`uu.msc.ai.mair.dialog.core.splits.get_split("original" | "grouped")` returns
`(x_train, x_test, y_train, y_test)` as lists of lower-cased strings, and
`uu.msc.ai.mair.dialog.metrics.evaluation` provides accuracy, balanced
accuracy, macro F1, the confusion matrix and the most frequent errors. Every
classifier should be trained and evaluated on both splits with these helpers
so that results are comparable.
