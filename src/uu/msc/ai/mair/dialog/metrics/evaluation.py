"""Shared evaluation utilities for all dialog act classifiers."""

from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Return accuracy, balanced accuracy and macro F1 as a dictionary."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def format_confusion_matrix(y_true: list[str], y_pred: list[str]) -> str:
    """Render the confusion matrix (rows = true, columns = predicted) as text."""
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    header = " " * 10 + " ".join(f"{label[:6]:>6}" for label in labels)
    rows = [f"{label:>10} " + " ".join(f"{n:>6}" for n in row)
            for label, row in zip(labels, matrix)]
    return "\n".join([header, *rows])


def print_evaluation(y_true: list[str], y_pred: list[str], title: str = "",
                     show_report: bool = True) -> dict[str, float]:
    """Print metrics and, optionally, the per-class report and confusion matrix."""
    metrics = compute_metrics(y_true, y_pred)
    if title:
        print(f"=== {title} ===")
    for name, value in metrics.items():
        print(f"{name:>18}: {value:.4f}")
    if show_report:
        labels = sorted(set(y_true) | set(y_pred))
        print()
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        print("confusion matrix (rows = true, columns = predicted):")
        print(format_confusion_matrix(y_true, y_pred))
    return metrics


def most_common_errors(x: list[str], y_true: list[str], y_pred: list[str],
                       top: int = 20) -> list[tuple[tuple[str, str, str], int]]:
    """Return the most frequent (utterance, true, predicted) misclassifications."""
    errors = Counter(
        (utterance, true, pred)
        for utterance, true, pred in zip(x, y_true, y_pred)
        if true != pred
    )
    return errors.most_common(top)
