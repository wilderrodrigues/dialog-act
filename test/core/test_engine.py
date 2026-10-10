import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from uu.msc.ai.mair.dialog.core.engine import calculate_loss_accuracy, evaluate_model


class FixedPredictionModel(nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return F.one_hot(batch, num_classes=3).to(torch.float)


def test_evaluate_model_keeps_classes_missing_from_evaluation_data() -> None:
    loader = _loader_without_class_two()

    _, confusion_matrix = evaluate_model(
        model=FixedPredictionModel(),
        dataset_loader=loader,
        device=torch.device("cpu"),
        mode="Test",
    )

    _assert_empty_class_is_present(confusion_matrix)


def test_calculate_loss_accuracy_keeps_classes_missing_from_validation_data() -> None:
    loader = _loader_without_class_two()

    _, confusion_matrix = calculate_loss_accuracy(
        model=FixedPredictionModel(),
        loss_fn=nn.CrossEntropyLoss(),
        dataset_loader=loader,
        device=torch.device("cpu"),
        mode="Validation",
    )

    _assert_empty_class_is_present(confusion_matrix)


def _loader_without_class_two() -> DataLoader:
    classes = torch.tensor([0, 1], dtype=torch.long)
    targets = F.one_hot(classes, num_classes=3).to(torch.float)
    return DataLoader(TensorDataset(classes, targets), batch_size=2)


def _assert_empty_class_is_present(confusion_matrix: np.ndarray) -> None:
    assert confusion_matrix.shape == (3, 3)
    assert confusion_matrix[2].sum() == 0
    assert confusion_matrix[:, 2].sum() == 0
