# Dialog Acts Classification

The project is based on the [Dialog Acts Classification](https://github.com/facebookresearch/ParlAI/tree/main/projects/dialog_acts)
project from the ParlAI repository.

## Usage

```sh
poetry run dialog-acts --help
```

# Baseline Model

```sh
poetry run dialog-acts eval-rule ./data/dialog_acts.dat
poetry run dialog-acts eval-rule ./data/dialog_acts.dat --split-strategy vanilla
poetry run dialog-acts eval-rule ./data/dialog_acts.dat --split-strategy grouped
```

# Logistic Regression

We are still working on the refactoring of the Logistic Regression model. However, the current implementation
under [Logistic Regression](./src/uu/msc/ai/mair/dialog/model/log_reg_classifier.py) is functional, with accuracy of
80.60%.

# Training the Neural Network model

The simplest way to train the model is to run:

```sh
poetry run dialog-acts train-nn ./data/dialog_acts.dat
```

Other parameters can be passed to the script. For example, one can decide to use a grouped split instead of a random one.
In addition to that, one can also specify the type of encoder to be used.

To see all options, please run:

```sh
poetry run dialog-acts train-nn --help
```

Currently, the Neural Network can be trained with the following flavours:

```sh
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder simple --epochs 5 --split-strategy vanilla
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder simple --epochs 5 --split-strategy grouped
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder bert --epochs 5 --split-strategy vanilla
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder bert --epochs 5 --split-strategy grouped
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder finetune --epochs 5 --split-strategy vanilla
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder finetune --epochs 5 --split-strategy grouped
```

The number of epochs given above is just for the sake of demonstration. The default value is set to 20. 

## Model checkpoints

The best model is saved under `ouput/best_model_epoch_N_YYYYMMD.pth`.

## Evaluating a trained Neural Network model

To evaluate a model, you will need a test dataset, which is a held-out split not present during training, and a
trained model. Please make sure to use either a `simple` or a `bert` encoder based model.

The commands below will evaluate the model on the test dataset:

```sh
poetry run dialog-acts eval-nn my-test-split.dat my-simple-encoder-model.pth --encoder simple
poetry run dialog-acts eval-nn my-test-split.dat my-bert-encoder-model.pth --encoder bert
```

The metrics will be printed out on the console, and the confusion matrix will be saved under `ouput/nn_confusion_matrix.png`.

# Frozen pretrained embeddings

[FrozenDistilBertEncoder](./src/uu/msc/ai/mair/dialog/core/encoders.py) turns utterances into features with a
frozen DistilBERT model. The pretrained weights are never updated: the model runs in evaluation mode inside
`torch.no_grad`, so it is a fixed feature extractor rather than something we train. It offers one shape per kind of
classifier, so every classifier can be trained on the same representation:
| method | shape | consumer |
|---|---|---|
| `encode_sentence` | `(n, MAX_TOKENS, 768)` | models that consume a sequence, such as the convolutional net |
| `encode_sentences` | `(n, 768)`, the `[CLS]` vector | models that need one fixed-size vector, such as logistic regression |


# Fine-tuned DistilBERT

With `--encoder finetune` the DistilBERT weights are trained end to end together with the classifier, instead of
being kept frozen:

```sh
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder finetune --epochs 20
```

