from pathlib import Path

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from uu.msc.ai.mair.dialog.core.datasets import DialogActsDataset, DatasetFactory
from uu.msc.ai.mair.dialog.core.engine import train_loop
from uu.msc.ai.mair.dialog.core.runtime import DeviceChoice, seed_everything, select_device
from uu.msc.ai.mair.dialog.model.nn_classifier import Conv1DClassifier

from typing import Annotated
import logging

import typer

app = typer.Typer(no_args_is_help=True, help="Dialog: Methods in AI Research.")
logging.basicConfig(level=logging.INFO)


@app.command(name="train", help="Train a model.")
def train(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    batch_size: Annotated[int, typer.Option(min=16, help="Batch size.")] = 64,
    epochs: Annotated[int, typer.Option(min=5, help="Maximum number of epochs.")] = 20,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 1e-3,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = DatasetFactory.VAL_SPLIT,
) -> None:
    seed_everything(seed=seed)

    utterances_train, utterances_val, acts_train, acts_val, targets = DatasetFactory.load_and_split_vanilla(data_path=dataset_path,
                                                                            split=val_split,
                                                                            seed=seed)
    dataset = DatasetFactory.load_dataframe(dataset_path, separator=" ")
    vocab = DatasetFactory.train_tokenizer(dataset=dataset)
    train_dataset = DialogActsDataset(vocab=vocab, utterances=utterances_train, acts=acts_train, targets=targets)
    val_dataset = DialogActsDataset(vocab=vocab, utterances=utterances_val, acts=acts_val, targets=targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    conv_classifier = Conv1DClassifier(vocabulary_size=len(vocab), n_classes=len(targets))
    optimizer = Adam(conv_classifier.parameters(), lr=lr)

    selected_device = select_device(device)
    train_loop(conv_classifier, loss_fn, optimizer, train_loader, val_loader, epochs, selected_device)


@app.command(name="eval", help="Evaluate a model.")
def evaluate() -> None:
    logging.info("This command is under development.")