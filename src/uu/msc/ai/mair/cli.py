from pathlib import Path

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
from uu.msc.ai.mair.dialog.model.nn_classifier import Conv1DClassifier, EmbeddingBagClassifier
from uu.msc.ai.mair.dialog.model.log_reg_classifier import LogRegModel
from uu.msc.ai.mair.dialog.model.rule_based_classifier import RuleBasedModel
from uu.msc.ai.mair.recommender.core.dialog_manager import DialogManager, Fallback, Mode

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score
import numpy as np


from typing import Annotated
import logging

import typer

from uu.msc.ai.mair.recommender.core.inference_engine import LogRegPredictor

app = typer.Typer(no_args_is_help=True, help="Dialog: Methods in AI Research.")
logging.basicConfig(level=logging.INFO)

nn_config = {"simple": SimpleEncoder, "bert": FrozenDistilBertEncoder, "finetune": DistilBertTokenEncoder,
             "vanilla": DatasetFactory.load_and_split_vanilla, "grouped": DatasetFactory.load_and_split_grouped}


def encode_one(encoder: Encoder, utterance: str, targets: dict[str, int]) -> torch.Tensor:
    dataset = DialogActsDataset(encoder=encoder, utterances=np.asarray([utterance]),
                                acts=np.asarray([next(iter(targets))]), targets=targets)
    features, _ = next(iter(DataLoader(dataset, batch_size=1)))
    return features


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

    split_strategy_loader = DatasetFactory.load_and_split_vanilla if split_strategy == "vanilla" else DatasetFactory.load_and_split_grouped
    utterances_train, utterances_val, acts_train, acts_val, targets = split_strategy_loader(data_path=dataset_path,
                                                                            split=val_split,
                                                                            seed=seed)

    dataset = DatasetFactory.load_dataframe(dataset_path, separator=separator)
    feature_encoder = nn_config[encoder]()
    feature_encoder.init_tokenizer(dataset)
    start_time = time.perf_counter()
    train_dataset = DialogActsDataset(encoder=feature_encoder, utterances=utterances_train, acts=acts_train, targets=targets)
    val_dataset = DialogActsDataset(encoder=feature_encoder, utterances=utterances_val, acts=acts_val, targets=targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    vocab_size = len(feature_encoder.get_vocabulary())

    if isinstance(feature_encoder, SimpleEncoder):
        padding_idx = feature_encoder.get_vocabulary()["[PAD]"]
        conv_classifier = EmbeddingBagClassifier(vocabulary_size=vocab_size, n_classes=len(targets),
                                                  padding_idx=padding_idx)
    elif isinstance(feature_encoder, DistilBertTokenEncoder):
        conv_classifier = FineTunedDistilBertClassifier(vocabulary_size=vocab_size, n_classes=len(targets),
                                                        max_tokens=feature_encoder.max_tokens)
    elif isinstance(feature_encoder, FrozenDistilBertEncoder):
        conv_classifier = Conv1DClassifier(vocabulary_size=vocab_size, n_classes=len(targets),
                                           max_tokens=feature_encoder.max_tokens, embed_len=EMBED_LEN)
        conv_classifier.embedding_layer = nn.Identity()
    else:
        raise ValueError("Invalid encoder type.")

    if isinstance(conv_classifier, FineTunedDistilBertClassifier):
        optimizer = Adam([{"params": conv_classifier.encoder_parameters(), "lr": encoder_lr},
                          {"params": conv_classifier.head_parameters(), "lr": lr}])
    else:
        optimizer = Adam(conv_classifier.parameters(), lr=lr)

    n_parameters = sum(parameter.numel() for parameter in conv_classifier.parameters())
    logging.info(f"Trainable parameters : {n_parameters:,}")

    output_dir = get_root() / "output_nn"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"nn_{encoder}_{split_strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = select_device(device)
    balanced_accuracy, confusion_matrix = train_loop(conv_classifier, loss_fn, optimizer, train_loader, val_loader,
                                                     epochs, selected_device, output_dir, model_name)
    logging.info(f"Training time : {time.perf_counter() - start_time:.1f} s")
    plot_confusion_matrix(confusion_matrix, targets, output_dir / "nn_confusion_matrix.png")

@app.command(name="eval-nn", help="Evaluate a model.")
def evaluate_nn(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the test dataset.")],
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    batch_size: Annotated[int, typer.Option(min=16, help="Batch size.")] = 64,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
    encoder: Annotated[str, typer.Option(help="Encoder to use when tokenizing the data. Must be 'simple', 'bert' or 'finetune'")] = "simple",) -> None:

    if encoder.lower() not in ("simple", "bert", "finetune"):
        raise typer.BadParameter(f"Invalid encoder: {encoder}. Must be one of 'simple', 'bert' or 'finetune'.")

    seed_everything(seed=seed)

    dialog_model = torch.load(model_path, weights_only=False, map_location=select_device(device))
    dialog_model.eval()

    utterances_test, acts_test, targets = DatasetFactory.load_test_dataset(data_path=dataset_path)

    dataset = DatasetFactory.load_dataframe(dataset_path, separator=separator)
    feature_encoder = nn_config[encoder]()
    feature_encoder.init_tokenizer(dataset)
    test_dataset = DialogActsDataset(encoder=feature_encoder, utterances=utterances_test, acts=acts_test, targets=targets)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    balanced_accuracy, confusion_matrix  = evaluate_model(model=dialog_model, dataset_loader=test_loader, device=select_device(device), mode="Test")
    output_dir = get_root() / "output_nn"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(confusion_matrix, targets, output_dir / "nn_confusion_matrix.png")


@app.command(name="prompt-nn", help="Classify typed utterances until 'exit' is entered.")
def prompt_nn(
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    dataset_path: Annotated[Path, typer.Argument(help="The dataset the model was trained on.")],
    device: Annotated[DeviceChoice, typer.Option(help="PyTorch compute device.")] = DeviceChoice.AUTO,
    encoder: Annotated[str, typer.Option(help="Must match training: 'simple', 'bert' or 'finetune'")] = "simple",
) -> None:
    if encoder.lower() not in ("simple", "bert", "finetune"):
        raise typer.BadParameter(f"Invalid encoder: {encoder}. Must be one of 'simple', 'bert' or 'finetune'.")

    selected_device = select_device(device)
    model = torch.load(model_path, weights_only=False, map_location=selected_device)
    model.eval()
    *_, targets = DatasetFactory.load_and_split_vanilla(data_path=dataset_path)
    acts = {idx: act for act, idx in targets.items()}
    feature_encoder = nn_config[encoder.lower()]()
    feature_encoder.init_tokenizer(DatasetFactory.load_dataframe(dataset_path, separator=" "))

    logging.info("Type an utterance to classify it, or 'exit' to quit.")
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
            index = model(encode_one(feature_encoder, utterance, targets).to(selected_device)).argmax(dim=1).item()
        logging.info(acts[index])


@app.command(name="train-logreg", help="Train a Logistic Regression model")
def train_log_reg(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    num_iterations: Annotated[int, typer.Option(min=100, help="Maximum number of iterations.")] = 1000,
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    val_split: Annotated[float, typer.Option(help="Validation split ratio.")] = DatasetFactory.VAL_SPLIT,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
    split_strategy: Annotated[str, typer.Option(help="Must be 'vanilla' or 'grouped'")] = "vanilla",
    encoder: Annotated[str, typer.Option(help="Must be 'simple' - for bag-of-words - or 'bert' (frozen DistilBERT)")] = "simple",
) -> None:
    if split_strategy.lower() not in ("vanilla", "grouped"):
        raise typer.BadParameter(f"Invalid split strategy: {split_strategy}. Must be one of 'vanilla' or 'grouped'.")
    if encoder.lower() not in ("simple", "bert"):
        raise ValueError(f"Invalid encoder: {encoder}. Must be 'simple' or 'bert'.")

    seed_everything(seed=seed)

    logreg = LogRegModel(dataset_path=dataset_path, separator=separator, num_iterations=num_iterations, encoder=encoder.lower())
    acts_val, acts_pred = logreg.train(split=val_split, seed=seed, split_strategy=split_strategy.lower())

    output_dir = get_root() / "output_log_reg"
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"log_reg_{encoder.lower()}_{split_strategy}"
    with open(output_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(logreg, f)

    accuracy = accuracy_score(acts_val, acts_pred)
    balanced_accuracy = balanced_accuracy_score(acts_val, acts_pred)

    recall = recall_score(acts_val, acts_pred, average="micro", labels=np.unique(acts_pred))
    precision = precision_score(acts_val, acts_pred, average="micro",
                                labels=np.unique(acts_pred))
    logging.info(f"Validation Accuracy: {accuracy:.3f}")
    logging.info(f"Validation Balanced Accuracy: {balanced_accuracy:.3f}")
    logging.info(f"Validation Recall: {recall:.3f}")
    logging.info(f"Validation Precision: {precision:.3f}")

    labels = list(logreg.targets)
    conf_matrix = confusion_matrix(acts_val, acts_pred, labels=labels)
    plot_confusion_matrix(conf_matrix, logreg.targets, output_dir / f"{name}_confusion_matrix.png")


@app.command(name="eval-logreg", help="Train a Logistic Regression model")
def evaluate_logreg(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    model_path: Annotated[Path, typer.Argument(help="Path to the trained NN model.")],
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the training.")] = DatasetFactory.SEED,
    encoder: Annotated[str, typer.Option(help="Must match training: 'simple' for bow or 'finetune'")] = "simple",
) -> None:
    if encoder.lower() not in ("simple", "bert"):
        raise ValueError(f"Invalid encoder: {encoder}. Must be 'simple' or 'bert'.")

    seed_everything(seed=seed)

    with open(model_path, 'rb') as file:
        log_reg = pickle.load(file)

    mode="Test"
    utterances_test, acts_test, targets = DatasetFactory.load_test_dataset(data_path=dataset_path)

    if encoder.lower() == "simple":
        vectorized_test_utterances = utterances_test
    elif encoder.lower() == "bert":
        frozen = nn_config["bert"]()
        frozen.init_tokenizer(None)
        vectorized_test_utterances = frozen.encode_sentences(list(utterances_test))

    predictions_test_set = log_reg.predict(vectorized_test_utterances)

    accuracy = accuracy_score(acts_test, predictions_test_set)
    balanced_accuracy = balanced_accuracy_score(acts_test, predictions_test_set)

    recall = recall_score(acts_test, predictions_test_set, average="micro", labels=np.unique(predictions_test_set))
    precision = precision_score(acts_test, predictions_test_set, average="micro", labels=np.unique(predictions_test_set))
    logging.info(f"{mode} Accuracy  : {accuracy:.3f}")
    logging.info(f"{mode} Balanced Accuracy  : {balanced_accuracy:.3f}")
    logging.info(f"{mode} Recall  : {recall:.3f}")
    logging.info(f"{mode} Precision  : {precision:.3f}")

    output_dir = get_root() / "output_log_reg"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(log_reg.targets)
    conf_matrix = confusion_matrix(acts_test, predictions_test_set, labels=labels)
    plot_confusion_matrix(conf_matrix, log_reg.targets, output_dir / "rule_based_confusion_matrix.png")


@app.command(name="rule-based", help="Evaluate the rule-based baseline.")
def rule_based(
    dataset_path: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    mode: Annotated[str, typer.Option(help="Evaluate the rule based model. The mode is either 'eval' or 'test'.")] = "eval",
    split_strategy: Annotated[str, typer.Option(help="Evaluate on the validation part of a 'vanilla' or 'grouped' split instead of the whole file.")] = "vanilla",
    seed: Annotated[int, typer.Option(min=0, help="Random seed for the split.")] = DatasetFactory.SEED,
    val_split: Annotated[float, typer.Option(help="Validation split ration.")] = DatasetFactory.VAL_SPLIT,
    separator: Annotated[str, typer.Option(help="DAT file separator.")] = " ",
) -> None:
    if mode.lower() == "test":
        utterances, acts, targets = DatasetFactory.load_test_dataset(data_path=dataset_path, separator=separator)
    elif mode.lower() == "eval":
        if split_strategy.lower() in ("vanilla", "grouped"):
            _, utterances, _, acts, targets = nn_config[split_strategy](data_path=dataset_path, separator=separator,
                                                                        split=val_split, seed=seed)
        else:
            raise typer.BadParameter(f"Invalid split strategy: {split_strategy}. Must be one of 'vanilla' or 'grouped'.")
    else:
        raise typer.BadParameter(f"Invalid mode type: {mode}. Must be one of 'eval' or 'test'.")

    predictions = RuleBasedModel().predict(utterances)
    recall = recall_score(acts, predictions, average="micro", labels=np.unique(predictions))
    precision = precision_score(acts, predictions, average="micro", labels=np.unique(predictions))
    logging.info(f"{mode} Accuracy  : {accuracy_score(acts, predictions):.3f}")
    logging.info(f"{mode} Balanced Accuracy  : {balanced_accuracy_score(acts, predictions):.3f}")
    logging.info(f"{mode} Recall  : {recall:.3f}")
    logging.info(f"{mode} Precision  : {precision:.3f}")

    output_dir = get_root() / "output_rube_based"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = list(targets)
    conf_matrix = confusion_matrix(acts, predictions, labels=labels)
    plot_confusion_matrix(conf_matrix, targets, output_dir / "rule_based_confusion_matrix.png")


@app.command(name="chat", help="Talk to the restaurant recommender, using the rule-based dialog act classifier.")
def chat(
    restaurants_path: Annotated[Path, typer.Option(help="Path to restaurant_info_extended.csv.")] = get_root() / "data" / "restaurant_info_extended.csv",
    model_type: Annotated[str, typer.Option(help="Model to be loaded and used by the dialog manager. Values should be 'logreg' or 'ann")] = "logreg",
    templates_path: Annotated[Path, typer.Option(help="Path to the response templates.")] = get_root() / "data" / "templates.json",
    seed: Annotated[int | None, typer.Option(help="Random seed for picking among matching restaurants.")] = DatasetFactory.SEED,
    show_acts: Annotated[bool, typer.Option(help="Print the classified act and dialog state after every turn.")] = False,
    log_dir: Annotated[Path, typer.Option(help="Directory for the JSON interaction logs.")] = get_root() / "output" / "dialogs",
    transparent: Annotated[bool, typer.Option(help="Explain the reasoning behind a recommendation.")] = True,
    fallback: Annotated[Fallback, typer.Option(help="Slot extraction when no keyword matches.")] = Fallback.LEVENSHTEIN,
    mode: Annotated[str, typer.Option(help="Dialog mode. 'patient' or 'eager'.")] = "patient"
) -> None:
    if model_type.lower() not in ("logreg", "ann"):
        raise ValueError(f"Invalid model: {model_type}. Must be 'logreg' or 'ann'.")

    dialog_predictor = None
    if model_type.lower() == "logreg":
        model_path = get_root() / "model-zoo" / "Logistic regression" / "log_reg_simple_grouped.pkl"
        dialog_predictor = LogRegPredictor(model_path)
        logging.info(f"Model type '{model_type}' loaded.")
    elif model_type.lower() == "ann":
        raise NotImplementedError("The Neural Network classifier is not yet implemented in the recommender yet.")

    seed_everything(seed=seed)
    DialogManager(classifier=dialog_predictor, restaurants_path=restaurants_path,
                  templates_path=templates_path, transparent=transparent, fallback = fallback, mode = Mode(mode)).run(log_dir=log_dir, show_acts=show_acts)
