from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import pickle
import time

from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DialogActsDataset, DatasetFactory
from uu.msc.ai.mair.dialog.core.encoders import (EMBED_LEN, Encoder, SimpleEncoder, FrozenDistilBertEncoder,
                                                 DistilBertTokenEncoder)
from uu.msc.ai.mair.dialog.core.engine import train_loop, evaluate_model
from uu.msc.ai.mair.dialog.core.runtime import DeviceChoice, seed_everything, select_device
from uu.msc.ai.mair.dialog.metrics.plot.utils import plot_confusion_matrix
from uu.msc.ai.mair.dialog.model.finetuned_classifier import FineTunedDistilBertClassifier
from uu.msc.ai.mair.dialog.model.nn_classifier import Conv1DClassifier
from uu.msc.ai.mair.dialog.model.log_reg_classifier import LogRegModel

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


from typing import Annotated
import logging

import typer

app = typer.Typer(no_args_is_help=True, help="Dialog: Methods in AI Research.")
logging.basicConfig(level=logging.INFO)

nn_config = {"simple": SimpleEncoder, "bert": FrozenDistilBertEncoder, "finetune": DistilBertTokenEncoder,
             "vanilla": DatasetFactory.load_and_split_vanilla, "grouped": DatasetFactory.load_and_split_grouped}


@app.command(name="train-nn", help="Train a Neural Network model.")
def train_nn(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    batch_size: Annotated[int, typer.Option(min=16, help="Batch size.")] = 64,
    epochs: Annotated[int, typer.Option(min=5, help="Maximum number of epochs.")] = 20,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 1e-3,
    encoder_lr: Annotated[float, typer.Option(help="Learning rate for the DistilBERT weights, 'finetune' encoder only.")] = 2e-5,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = DatasetFactory.VAL_SPLIT,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
    encoder: Annotated[str, typer.Option(help="Encoder to use when tokenizing the data. Must be 'simple', 'bert' or 'finetune'")] = "simple",
    split_strategy: Annotated[str, typer.Option(help="Split strategy to be used. Must be 'vanilla' or 'grouped'")] = "vanilla",
) -> None:
    seed_everything(seed=seed)

    if split_strategy not in nn_config:
        raise ValueError(f"Invalid split strategy: {split_strategy}. Must be one of 'vanilla' or 'grouped'.'")

    if encoder not in nn_config:
        raise ValueError(f"Invalid encoder: {encoder}. Must be one of 'simple', 'bert' or 'finetune'.'")

    split_strategy = DatasetFactory.load_and_split_vanilla if split_strategy == "vanilla" else DatasetFactory.load_and_split_grouped
    utterances_train, utterances_val, acts_train, acts_val, targets = split_strategy(data_path=dataset_path,
                                                                            split=val_split,
                                                                            seed=seed)

    dataset = DatasetFactory.load_dataframe(dataset_path, separator=separator)
    encoder = nn_config[encoder]()
    encoder.init_tokenizer(dataset)
    start_time = time.perf_counter()
    train_dataset = DialogActsDataset(encoder=encoder, utterances=utterances_train, acts=acts_train, targets=targets)
    val_dataset = DialogActsDataset(encoder=encoder, utterances=utterances_val, acts=acts_val, targets=targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    vocab_size = len(encoder.get_vocabulary())

    if isinstance(encoder, SimpleEncoder):
        conv_classifier = Conv1DClassifier(vocabulary_size=vocab_size, n_classes=len(targets))
    elif isinstance(encoder, FrozenDistilBertEncoder):
        conv_classifier = Conv1DClassifier(vocabulary_size=vocab_size, n_classes=len(targets),
                                           max_tokens=encoder.max_tokens, embed_len=EMBED_LEN)
        conv_classifier.embedding_layer = nn.Identity()
    elif isinstance(encoder, DistilBertTokenEncoder):
        conv_classifier = FineTunedDistilBertClassifier(vocabulary_size=vocab_size, n_classes=len(targets),
                                                        max_tokens=encoder.max_tokens)
    else:
        raise ValueError("Invalid encoder type.")

    if isinstance(conv_classifier, FineTunedDistilBertClassifier):
        optimizer = Adam([{"params": conv_classifier.encoder_parameters(), "lr": encoder_lr},
                          {"params": conv_classifier.head_parameters(), "lr": lr}])
    else:
        optimizer = Adam(conv_classifier.parameters(), lr=lr)

    n_parameters = sum(parameter.numel() for parameter in conv_classifier.parameters())
    logging.info(f"Trainable parameters : {n_parameters:,}")

    output_dir = get_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = select_device(device)
    balanced_accuracy, confusion_matrix = train_loop(conv_classifier, loss_fn, optimizer, train_loader, val_loader,
                                                     epochs, selected_device, output_dir)
    logging.info(f"Training time : {time.perf_counter() - start_time:.1f} s")
    plot_confusion_matrix(confusion_matrix, targets, output_dir)

@app.command(name="eval-nn", help="Evaluate a model.")
def evaluate_nn(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the test dataset.")],
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    batch_size: Annotated[int, typer.Option(min=16, help="Batch size.")] = 64,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
    encoder: Annotated[str, typer.Option(help="Encoder to use when tokenizing the data. Must be 'simple', 'bert' or 'finetune'")] = "simple",) -> None:

    seed_everything(seed=seed)

    dialog_model = torch.load(model_path, weights_only=False, map_location=select_device(device))
    dialog_model.eval()

    utterances_test, acts_test, targets = DatasetFactory.load_test_dataset(data_path=dataset_path)

    dataset = DatasetFactory.load_dataframe(dataset_path, separator=separator)
    encoder = nn_config[encoder]()
    encoder.init_tokenizer(dataset)
    test_dataset = DialogActsDataset(encoder=encoder, utterances=utterances_test, acts=acts_test, targets=targets)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    balanced_accuracy, confusion_matrix  = evaluate_model(model=dialog_model, dataset_loader=test_loader, device=select_device(device), mode="Test")
    output_dir = get_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(confusion_matrix, targets, output_dir)

def encode_one(encoder: Encoder, utterance: str, targets: dict[str, int]) -> torch.Tensor:
    # Reuses the training pipeline, but dataset need a label per utterance.
    dataset = DialogActsDataset(encoder=encoder, utterances=np.asarray([utterance]),
                                acts=np.asarray([next(iter(targets))]), targets=targets)
    features, _ = next(iter(DataLoader(dataset, batch_size=1)))
    return features


@app.command(name="prompt-nn", help="Classify typed utterances until 'exit' is entered.")
def prompt_nn(
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    dataset_path: Annotated[Path, typer.Argument(help="The dataset the model was trained on.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    encoder: Annotated[str, typer.Option(help="Must match training: 'simple', 'bert' or 'finetune'")] = "simple",
) -> None:
    if encoder not in ("simple", "bert", "finetune"):
        raise typer.BadParameter(f"Invalid encoder: {encoder}. Must be one of 'simple', 'bert' or 'finetune'.")

    selected_device = select_device(device)
    model = torch.load(model_path, weights_only=False, map_location=selected_device)
    model.eval()
    *_, targets = DatasetFactory.load_and_split_vanilla(data_path=dataset_path)
    acts = {idx: act for act, idx in targets.items()}
    encoder = nn_config[encoder]()
    encoder.init_tokenizer(DatasetFactory.load_dataframe(dataset_path, separator=" "))

    print("Type an utterance to classify it, or 'exit' to quit.")
    while True:
        try:
            utterance = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if utterance in ("exit", "quit"):
            break
        if not utterance:
            continue

        with torch.no_grad():
            index = model(encode_one(encoder, utterance, targets).to(selected_device)).argmax(dim=1).item()
        print(acts[index])




@app.command(name="train-logreg", help="Train a Logistic Regression model")
def train_log_reg(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    num_iterations: Annotated[int, typer.Option(min=100, help="Maximum number of iterations.")] = 1000,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = DatasetFactory.VAL_SPLIT,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
) -> None:
    seed_everything(seed=seed)

    logreg = LogRegModel(dataset_path,num_iterations=num_iterations)
    acts_val, acts_pred = logreg.train(split=val_split,seed=seed)


    output_dir = get_root() / "output_log_reg"
    output_dir.mkdir(parents=True, exist_ok=True)


    with open(output_dir / "log_reg", "wb") as f:
        pickle.dump(logreg, f)
    labels = list(logreg.targets)
    conf_matrix = confusion_matrix(acts_val, acts_pred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logreg.targets)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "log_reg_confusion_matrix.png")
    plt.show()


@app.command(name="evaluate-log-reg", help="Train a Logistic Regression model")
def evaluate_log_reg(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = DatasetFactory.VAL_SPLIT,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
) -> None:
    seed_everything(seed=seed)

    with open(model_path, 'rb') as file:
        dialog_model_log_reg = pickle.load(file)

    mode="Test"
    utterances_test, acts_test, targets = DatasetFactory.load_test_dataset(data_path=dataset_path)
    predictions_test_set = dialog_model_log_reg.predict(utterances_test)

    accuracy = accuracy_score(acts_test, predictions_test_set)
    balanced_accuracy = balanced_accuracy_score(acts_test, predictions_test_set)

    conf_matrix = confusion_matrix(acts_test, predictions_test_set)
    recall = recall_score(acts_test, predictions_test_set, average="micro", labels=np.unique(predictions_test_set))
    precision = precision_score(acts_test, predictions_test_set, average="micro", labels=np.unique(predictions_test_set))
    logging.info(f"{mode} Accuracy  : {accuracy:.3f}")
    logging.info(f"{mode} Balanced Accuracy  : {balanced_accuracy:.3f}")
    logging.info(f"{mode} Recall  : {recall:.3f}")
    logging.info(f"{mode} Precision  : {precision:.3f}")
    conf_matrix = confusion_matrix(acts_test, predictions_test_set)

    output_dir = get_root() / "output_log_reg"
    output_dir.mkdir(parents=True, exist_ok=True)


    labels = list(dialog_model_log_reg.targets)
    conf_matrix = confusion_matrix(acts_test, predictions_test_set,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "eval_log_reg_confusion_matrix.png")
    plt.show()




