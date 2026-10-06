# Methods in Artificial Intelligence Project

This project is a part of the course Methods in Artificial Intelligence. It covers the implementation
of several Dialog Acts classifiers, where we explored the following: Simple rule-base classifier; a vanilla bag-of-words
classifier; a 1D Convolutional Neural Network, using bag-of-words features, classifier; and DistilBert classifier.

Moreover, we explored the use of a pre-trained DistilBert model to finetune the 1D CNN classifier.

## Setup

Install Poetry 2.1+ and Python 3.13.15.

To install PyEnv and Poetry, if necessary, please follow the official documentation:

* [Poetry](https://python-poetry.org/docs/).
* [PyEnv](https://github.com/pyenv/pyenv#installation).

```sh
pyenv install -s 3.13.15
pyenv local 3.13.15 # or local, if you do not want to mess with your environment. 
poetry env use "$(pyenv which python)"
poetry install
```

Run these commands from the project directory; `.python-version` selects the
project interpreter. Poetry creates the environment in `.venv`. No global
Python selection is required. Commit `poetry.lock` with the project.

