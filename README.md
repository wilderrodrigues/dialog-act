# Methods in Artificial Intelligence Project

This project is a part of the course Methods in Artificial Intelligence. It covers the implementation
of several Dialog Acts classifiers, where we explored the following: Simple rule-base classifier; a vanilla bag-of-words
classifier; a 1D Convolutional Neural Network, using bag-of-words features, classifier; and DistilBert classifier.

Moreover, we explored the use of a pre-trained DistilBert model to finetune the 1D CNN classifier.

## Setup

To be able to run the project, you will need Python installed in your operating system. Any version in the range `>=3.13.15,<3.14`
will suffice.

Once your Python installation is ready, please make sure you have the latest version of Poetry installed.

### Poetry for Windows

The installer script is available directly at install.python-poetry.org, and is developed in its own repository.
The script can be executed directly (i.e. ‘curl python’) or downloaded and then executed from disk (e.g. in a CI environment).

### Linux, macOS, Windows (WSL)

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

### Windows (PowerShell)

```sh
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

If you run into issues, please refer to the official installation instructions: https://python-poetry.org/docs/#installation

## Project Installation

Once Python and Poetry are installed, please install the project dependencies. The command must be executed from the project root directory.

```sh
poetry install
```

### Unit Tests

To check that the delivered code is working as expected, yoou can run the unit tests:

```sh
poetry run pytest
```

## Project Structure

We have split the project into two parts: `dialog`; and `recommender`. You will find mode information under the
respective README files.

* [Dialog Acts](./src/uu/msc/ai/mair/dialog/README.md)
* [Recommender](./src/uu/msc/ai/mair/recommender/README.md)