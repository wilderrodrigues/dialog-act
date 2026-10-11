# Dialog Acts Classification

The project is based on the [Dialog Acts Classification](https://github.com/facebookresearch/ParlAI/tree/main/projects/dialog_acts)
project from the ParlAI repository.

## Usage

the `dialog-acts` CLI (Command Line Interface) offers commands to train and evaluate diverse model types. Before
you try to train a model, you can pass the `--help` option to check all possible commands.

```sh
poetry run dialog-acts --help
```

## Training models

### Rule based model

To inspect the options available, please run the command below: 

```sh
poetry run dialog-acts rule-based --help
```

You can run the rule-based command with either a `vanilla` split strategy, which means we will take 85/15 ratio form the dataset
and use the 15% to evaluate the model, or you can use the `grouped` strategy, which will use a stratified grouped strategy taking
into account K fold splits, where K is equals to 7 (1 / 15 ~ 6.67).

```sh
poetry run dialog-acts rule-based ./data/dialog_acts.dat --split-strategy vanilla
poetry run dialog-acts rule-ruled ./data/dialog_acts.dat --split-strategy grouped
```

#### Testing the Rele Based

The rule based can be tested with a held-out test set using the following command:

```sh
poetry run dialog-acts rule-based ./data/dialog_acts.dat --mode test
```

### Logistic Regression

The Logistic Regression model has 4 flavours:

* Vanilla splits (85/15 ratio) with Bag-of-Words strategy;
* Stratified grouped splits with Bag-of-Words strategy;
* Vanilla splits with BERT embeddings strategy; and
* Stratified grouped splits with BERT embeddings strategy.

To inspect the options available, please run the command below: 

```sh
poetry run dialog-acts train-logreg --help
```

One can train a Logistic Regression model by running one of the commands below:

```sh
poetry run dialog-acts train-logreg ./data/dialog_acts.dat --encoder simple --split-strategy vanilla
poetry run dialog-acts train-logreg ./data/dialog_acts.dat --encoder simple --split-strategy grouped
poetry run dialog-acts train-logreg ./data/dialog_acts.dat --encoder bert --split-strategy vanilla
poetry run dialog-acts train-logreg ./data/dialog_acts.dat --encoder bert --split-strategy grouped
```

### Neural Network model

The Neural Networks available have more flavours than the Logistic Regression model. In total, we can train 6
different models using a variety of architectures and strategies. The available architectures are the following:

* Embedding Bag-of-Words using a linear layers;
* Vanilla embedding layer, followed by 1 dimension convolution layers and a linear output layer;
* Frozen Distil BERT encoder used to bootstrap the Conv1D network; and
* A full distil BERT encoder, unfrozen, used along side the Conv1D network.

In terms of training strategy, we use the same methods as with the Logistic Regression, plus 2 new ones based on the
unfrozen model:

* Vanilla splits (85/15 ratio) with Bag-of-Words strategy;
* Stratified grouped splits with Bag-of-Words strategy;
* Vanilla splits with BERT embeddings strategy;
* Stratified grouped splits BERT embeddings strategy;
* Vanilla splits with unfrozen BERT model strategy;
* Stratified grouped splits with unfrozen BERT model strategy.

To inspect the options available, please run the command below: 

```sh
poetry run dialog-acts train-nn --help
```

One can train a Neural Network model by running one of the commands below:

```sh
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder simple --epochs 10 --split-strategy vanilla
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder simple --epochs 10 --split-strategy grouped
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder bert --epochs 10 --split-strategy vanilla
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder bert --epochs 10 --split-strategy grouped
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder finetune --epochs 10 --split-strategy vanilla
poetry run dialog-acts train-nn ./data/dialog_acts.dat --encoder finetune --epochs 10 --split-strategy grouped
```

The unfrozen BERT models has 66 million parameters. So, training it needs more time. On a MacBook Pro M4 using the
MPS device, it took 2 minutes per epoch.

The number of epochs given above is just for the sake of demonstration. The default value is set to 20. 

## Model checkpoints

We also pushed a Model Zoo containing 4 Logistic Regression and 4 Neural Network models. We have not pushed the unfrozen 
BERT models because those were too big: 250MB. The Model Zoo artefacts can be found under the directory `model-zoo`.

If you train you own models, for more epochs for example, the resulting artefacts will be saved under `output_log_reg` and `output_nn`
for the Logistic Regression and Neural Networks, respectively.

## Evaluating models

Both Logistic Regression and Neural Network models have an evaluation command. Below we show two simple ways to validate
either model. It's important to note that although we are using the training dataset in the examples below, you should use
a held-out test set with those commands.

* Logistic Regression
```sh
poetry run dialog-acts eval-logreg \
                       ./data/dialog_acts.dat \
                       ./model-zoo/Logistic\ regression/log_reg_simple_vanilla.pkl \
                       --encoder simple
```

* Neural Network 
```sh
poetry run dialog-acts eval-nn \
                       ./data/dialog_acts.dat \
                       ./model-zoo/Neural\ networks/nn-bow-vanilla.pth \
                       --encoder simple
```

The metrics will be printed out on the console, and the confusion matrix will be saved under the already indicated
output directories: `output_log_reg` and `output_nn`, for the Logistic Regression and Neural Netroks, respectively.

## Authors

The authors are depicted in alphabetical order and have equal contribution to the project.

* Cai Jiacheng
* Dexter Fijn van Draat
* Remy Jager
* Wilder C. Rodrigues
