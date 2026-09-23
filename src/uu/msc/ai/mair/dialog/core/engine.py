import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)

@torch.no_grad()
def calculate_loss_accuracy(model: torch.nn.Module, loss_fn: torch.nn.Module,
                            val_loader: DataLoader, device: torch.device, mode: str) -> None:
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


def train_loop(model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device):
    model.to(device)
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"Train Loss : {torch.tensor(losses).mean():.3f}")
        calculate_loss_accuracy(model, loss_fn, val_loader, device, "Validation")