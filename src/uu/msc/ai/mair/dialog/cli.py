from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DialogActsDataset, DatasetFactory
from uu.msc.ai.mair.dialog.core.encoders import EMBED_LEN, SimpleEncoder, FrozenDistilBertEncoder
from uu.msc.ai.mair.dialog.core.engine import train_loop, evaluate_model
from uu.msc.ai.mair.dialog.core.runtime import DeviceChoice, seed_everything, select_device
from uu.msc.ai.mair.dialog.metrics.plot.utils import plot_confusion_matrix
from uu.msc.ai.mair.dialog.model.nn_classifier import Conv1DClassifier

from typing import Annotated
import logging

import typer

app = typer.Typer(no_args_is_help=True, help="Dialog: Methods in AI Research.")
logging.basicConfig(level=logging.INFO)

nn_config = {"simple": SimpleEncoder, "bert": FrozenDistilBertEncoder,
             "vanilla": DatasetFactory.load_and_split_vanilla, "grouped": DatasetFactory.load_and_split_grouped}


@app.command(name="train-nn", help="Train a Neural Network model.")
def train_nn(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    batch_size: Annotated[int, typer.Option(min=16, help="Batch size.")] = 64,
    epochs: Annotated[int, typer.Option(min=5, help="Maximum number of epochs.")] = 20,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 1e-3,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = DatasetFactory.VAL_SPLIT,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
    encoder: Annotated[str, typer.Option(help="Encoder to use when tokenizing the data. Must be 'simple' or 'bert'")] = "simple",
    split_strategy: Annotated[str, typer.Option(help="Split strategy to be used. Must be 'vanilla' or 'grouped'")] = "vanilla",
) -> None:
    seed_everything(seed=seed)

    if split_strategy not in nn_config:
        raise ValueError(f"Invalid split strategy: {split_strategy}. Must be one of 'vanilla' or 'grouped'.'")

    if encoder not in nn_config:
        raise ValueError(f"Invalid encoder: {encoder}. Must be one of 'simple' or 'bert'.'")

    split_strategy = DatasetFactory.load_and_split_vanilla if split_strategy == "vanilla" else DatasetFactory.load_and_split_grouped
    utterances_train, utterances_val, acts_train, acts_val, targets = split_strategy(data_path=dataset_path,
                                                                            split=val_split,
                                                                            seed=seed)

    dataset = DatasetFactory.load_dataframe(dataset_path, separator=separator)
    encoder = nn_config[encoder]()
    encoder.init_tokenizer(dataset)
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
    else:
        raise ValueError("Invalid encoder type.")

    optimizer = Adam(conv_classifier.parameters(), lr=lr)

    output_dir = get_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = select_device(device)
    balanced_accuracy, confusion_matrix = train_loop(conv_classifier, loss_fn, optimizer, train_loader, val_loader,
                                                     epochs, selected_device, output_dir)
    plot_confusion_matrix(confusion_matrix, targets, output_dir)

@app.command(name="eval-nn", help="Evaluate a model.")
def evaluate_nn(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the test dataset.")],
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    batch_size: Annotated[int, typer.Option(min=16, help="Batch size.")] = 64,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
    encoder: Annotated[str, typer.Option(help="Encoder to use when tokenizing the data. Must be 'simple' or 'bert'")] = "simple",) -> None:

    seed_everything(seed=seed)

    dialog_model = torch.load(model_path, weights_only=False, map_location=select_device(device))
    dialog_model.eval()

    utterances_test, acts_test, targets = DatasetFactory.load_test_dataset(data_path=dataset_path, seed=seed)

    dataset = DatasetFactory.load_dataframe(dataset_path, separator=separator)
    encoder = nn_config[encoder]()
    encoder.init_tokenizer(dataset)
    test_dataset = DialogActsDataset(encoder=encoder, utterances=utterances_test, acts=acts_test, targets=targets)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    balanced_accuracy, confusion_matrix  = evaluate_model(model=dialog_model, dataset_loader=test_loader, device=select_device(device), mode="Test")
    output_dir = get_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(confusion_matrix, targets, output_dir)