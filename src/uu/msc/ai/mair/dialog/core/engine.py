import time
from pathlib import Path

import torch
import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score
import logging

logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataset_loader: DataLoader, device: torch.device, mode: str) -> tuple[float, npt.NDArray]:
    targets, predictions = [],[]
    for utterances_input, acts_output in dataset_loader:
        utterances_input = utterances_input.to(device)
        acts_output = acts_output.to(device)
        preds = model(utterances_input)

        targets.append(acts_output.argmax(dim=1))
        predictions.append(preds.argmax(dim=1))

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)

    detached_targets = targets.detach().cpu().numpy()
    detached_predictions = predictions.detach().cpu().numpy()
    accuracy = accuracy_score(detached_targets, detached_predictions)
    balanced_accuracy = balanced_accuracy_score(detached_targets, detached_predictions)

    conf_matrix = confusion_matrix(detached_targets, detached_predictions)
    recall = recall_score(detached_targets, detached_predictions, average="micro", labels=np.unique(detached_predictions))
    precision = precision_score(detached_targets, detached_predictions, average="micro", labels=np.unique(detached_predictions))

    logging.info(f"{mode} Accuracy  : {accuracy:.3f}")
    logging.info(f"{mode} Balanced Accuracy  : {balanced_accuracy:.3f}")
    logging.info(f"{mode} Recall  : {recall:.3f}")
    logging.info(f"{mode} Precision  : {precision:.3f}")

    return balanced_accuracy, conf_matrix

@torch.no_grad()
def calculate_loss_accuracy(model: torch.nn.Module, loss_fn: torch.nn.Module,
                            dataset_loader: DataLoader, device: torch.device, mode: str) -> tuple[float, npt.NDArray]:
    targets, predictions, losses = [],[],[]
    for utterances_input, acts_output in dataset_loader:
        utterances_input = utterances_input.to(device)
        acts_output = acts_output.to(device)
        preds = model(utterances_input)
        loss = loss_fn(preds, acts_output)
        losses.append(loss.item())

        targets.append(acts_output.argmax(dim=1))
        predictions.append(preds.argmax(dim=1))

    targets = torch.cat(targets)
    predictions = torch.cat(predictions)

    logging.info(f"{mode} Loss : {torch.tensor(losses).mean():.3f}")
    detached_targets = targets.detach().cpu().numpy()
    detached_predictions = predictions.detach().cpu().numpy()
    accuracy = accuracy_score(detached_targets, detached_predictions)
    balanced_accuracy = balanced_accuracy_score(detached_targets, detached_predictions)

    conf_matrix = confusion_matrix(detached_targets, detached_predictions)
    recall = recall_score(detached_targets, detached_predictions, average="micro", labels=np.unique(detached_predictions))
    precision = precision_score(detached_targets, detached_predictions, average="micro", labels=np.unique(detached_predictions))

    logging.info(f"{mode} Accuracy  : {accuracy:.3f}")
    logging.info(f"{mode} Balanced Accuracy  : {balanced_accuracy:.3f}")
    logging.info(f"{mode} Recall  : {recall:.3f}")
    logging.info(f"{mode} Precision  : {precision:.3f}")
    return balanced_accuracy, conf_matrix


def train_loop(model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device,
               output_dir: Path) -> tuple[float, npt.NDArray]:
    model.to(device)
    best_accuracy = 0.0
    conf_matrix = None
    for i in range(1, epochs+1):
        losses = []
        for utterances, acts in tqdm(train_loader):
            model.train()
            utterances = utterances.to(device)
            acts = acts.to(device)
            acts_pred = model(utterances)

            loss = loss_fn(acts_pred, acts)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Train Loss : {torch.tensor(losses).mean():.3f}")
        model.eval()
        epoch_balanced_accuracy, conf_matrix = calculate_loss_accuracy(model, loss_fn, val_loader, device, "Validation")
        if epoch_balanced_accuracy > best_accuracy:
            best_accuracy = epoch_balanced_accuracy
            model_name = f"best_model_epoch_{str(epochs)}_{time.strftime('%Y%m%d')}.pth"
            torch.save(model, output_dir / model_name)
            logging.info(f"Best Balanced Accuracy : {best_accuracy:.3f} saved as {model_name}.")

    return best_accuracy, conf_matrix
