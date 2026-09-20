from pathlib import Path

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from uu.msc.ai.mair.dialog.core.datasets import DialogActsDataset, DatasetFactory
from uu.msc.ai.mair.dialog.core.engine import train_loop
from uu.msc.ai.mair.dialog.core.runtime import DeviceChoice, seed_everything, select_device
from uu.msc.ai.mair.dialog.core.splits import DEFAULT_DATA_PATH, RANDOM_SEED, SPLIT_NAMES, get_split
from uu.msc.ai.mair.dialog.metrics.evaluation import most_common_errors, print_evaluation
from uu.msc.ai.mair.dialog.model import baseline
from uu.msc.ai.mair.dialog.model.classifier import Conv1DClassifier

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
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = 42,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 1e-3,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = 0.4,
    test_split: Annotated[float, typer.Option(help="Test split ration.")] = 0.6,
) -> None:
    seed_everything(seed=seed)

    train_split, val_split, test_split, targets = DatasetFactory.load_and_split_dataset(data_path=dataset_path,
                                                                                        separator=" ",
                                                                                        splits=(val_split, test_split))
    dataset = DatasetFactory.load_dataframe(dataset_path, separator=" ")
    vocab = DatasetFactory.train_tokenizer(dataset=dataset)
    train_dataset = DialogActsDataset(vocab=vocab, dataframe=train_split, targets=targets)
    val_dataset = DialogActsDataset(vocab=vocab, dataframe=val_split, targets=targets)
    test_dataset = DialogActsDataset(vocab=vocab, dataframe=test_split, targets=targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    conv_classifier = Conv1DClassifier(vocabulary_size=len(vocab), n_classes=len(targets))
    optimizer = Adam(conv_classifier.parameters(), lr=lr)

    selected_device = select_device(device)
    train_loop(conv_classifier, loss_fn, optimizer, train_loader, val_loader, test_loader, epochs, selected_device)


@app.command(name="eval", help="Evaluate a model.")
def evaluate() -> None:
    logging.info("This command is under development.")


@app.command(name="baseline", help="Evaluate the rule-based keyword baseline.")
def run_baseline(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset")] = DEFAULT_DATA_PATH,
    split: Annotated[str, typer.Option(help="Split to evaluate: original, grouped or both.")] = "both",
    errors: Annotated[int, typer.Option(min=0, help="Number of most frequent test errors to print.")] = 0,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the split.")] = RANDOM_SEED,
) -> None:
    split_names = SPLIT_NAMES if split == "both" else (split,)
    for split_name in split_names:
        x_train, x_test, y_train, y_test = get_split(split_name, dataset_path, seed)
        print_evaluation(y_train, baseline.predict(x_train),
                         f"baseline / {split_name} / train", show_report=False)
        print()
        y_pred = baseline.predict(x_test)
        print_evaluation(y_test, y_pred, f"baseline / {split_name} / test")
        if errors > 0:
            print()
            print("most frequent test errors (count  utterance | true -> predicted):")
            for (utterance, true, pred), count in most_common_errors(x_test, y_test, y_pred, errors):
                print(f"{count:>4}  {utterance!r} | {true} -> {pred}")
        print()


@app.command(name="predict", help="Classify utterances typed at a prompt until 'exit'.")
def predict_prompt() -> None:
    print("Type an utterance to classify it (empty line or 'exit' to quit).")
    while True:
        try:
            utterance = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if utterance.strip().lower() in ("", "exit", "quit"):
            break
        print(baseline.predict_one(utterance))
