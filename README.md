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

# Frozen pretrained embeddings

`core.embeddings.FrozenDistilBertEncoder` turns utterances into features with a
frozen DistilBERT model. The pretrained weights are never updated: the model
runs in evaluation mode inside `torch.no_grad`, so it is a fixed feature
extractor rather than something we train. It offers one shape per kind of
classifier, so every classifier can be trained on the same representation:

| method | shape | consumer |
|---|---|---|
| `encode_tokens` | `(n, 24, 768)` | models that consume a sequence, such as the convolutional net |
| `encode_sentences` | `(n, 768)` | models that need one fixed-size vector, such as logistic regression |

`encode_sentences` averages the token vectors over the real tokens only, so the
result does not depend on how long the other utterances in the batch happen to
be. In `encode_tokens` the padded positions are zeroed, so they cannot
contribute to a convolution. `MAX_TOKENS` is 24 because the longest utterance
in the data is 26 sub-word tokens and the 99th percentile is 16.

This needs the `transformers` package, which is not a project dependency yet:

```sh
poetry add transformers
```

```python
from uu.msc.ai.mair.dialog.core.embeddings import FrozenDistilBertEncoder

encoder = FrozenDistilBertEncoder()

sentence_features = encoder.encode_sentences(utterances)  # (n, 768)
token_features = encoder.encode_tokens(utterances)        # (n, 24, 768)
```

The first call downloads `distilbert-base-uncased` (about 260 MB) from the
Hugging Face hub. Encoding the full data set takes roughly ten seconds on a
GPU and a few minutes on CPU.

```sh
poetry run pytest test/core/test_embeddings.py
```
