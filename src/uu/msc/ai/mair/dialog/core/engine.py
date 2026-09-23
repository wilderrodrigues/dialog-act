import math

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import logging

from uu import get_root

logging.basicConfig(level=logging.INFO)

@torch.no_grad()
def calculate_loss_accuracy(model: torch.nn.Module, loss_fn: torch.nn.Module,
                            val_loader: DataLoader, device: torch.device, mode: str) -> float:
    targets, predictions, losses = [],[],[]
    for utterances_input, acts_output in val_loader:
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
    accuracy = accuracy_score(targets.detach().cpu().numpy(), predictions.detach().cpu().numpy())
    logging.info(f"{mode} Accuracy  : {accuracy:.3f}")
    return accuracy


def train_loop(model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device):
    output_dir = get_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    best_accuracy = 0.0
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
        epoch_accuracy = calculate_loss_accuracy(model, loss_fn, val_loader, device, "Validation")
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            model_name = f"best_model_epoch_{str(epochs)}_{time.strftime('%Y%m%d')}.pt"
            torch.save(model.state_dict(), output_dir / model_name)
            logging.info(f"Best Accuracy : {best_accuracy:.3f} saved as {model_name}.")
