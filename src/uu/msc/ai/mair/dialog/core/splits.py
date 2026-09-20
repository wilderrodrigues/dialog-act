"""Loading and splitting of the dialog act dataset (dialog_acts.dat).

Two split strategies are provided, both stratified on the dialog act:

* original: plain random 85/15 split; identical utterances may appear on
  both sides.
* grouped: 85/15 split where every unique utterance is assigned to exactly
  one side, so no identical utterance leaks from train into test.
"""

from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from uu import get_root

DEFAULT_DATA_PATH = get_root() / "data" / "dialog_acts.dat"

TEST_SIZE = 0.15
RANDOM_SEED = 42
SPLIT_NAMES = ("original", "grouped")

Split = tuple[list[str], list[str], list[str], list[str]]


def load_data(path: Path = DEFAULT_DATA_PATH) -> tuple[list[str], list[str]]:
    """Read a .dat file with lines 'dialog_act utterance' into (labels, utterances).

    Utterances are lower-cased. Empty lines are skipped; a line without an
    utterance keeps its label with an empty string.
    """
    labels: list[str] = []
    utterances: list[str] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            label, _, utterance = line.partition(" ")
            labels.append(label.lower())
            utterances.append(utterance.lower().strip())
    return labels, utterances


def original_split(labels: list[str], utterances: list[str], seed: int = RANDOM_SEED) -> Split:
    """Stratified random split; returns (x_train, x_test, y_train, y_test)."""
    x_train, x_test, y_train, y_test = train_test_split(
        utterances, labels, test_size=TEST_SIZE, stratify=labels, random_state=seed
    )
    return x_train, x_test, y_train, y_test


def grouped_split(labels: list[str], utterances: list[str], seed: int = RANDOM_SEED) -> Split:
    """Stratified split with all duplicates of an utterance on the same side.

    Every unique utterance forms one group. Groups are stratified on their
    most frequent label (a few utterances carry more than one label in the
    data), then all occurrences follow the group.
    """
    label_counts: dict[str, Counter] = {}
    for label, utterance in zip(labels, utterances):
        label_counts.setdefault(utterance, Counter())[label] += 1

    unique_utterances = list(label_counts)
    group_labels = [label_counts[u].most_common(1)[0][0] for u in unique_utterances]

    # Stratification needs at least two members per class; a class with a
    # single unique utterance is stratified together with the majority class.
    class_sizes = Counter(group_labels)
    majority = class_sizes.most_common(1)[0][0]
    strata = [g if class_sizes[g] > 1 else majority for g in group_labels]

    _, test_groups = train_test_split(
        unique_utterances, test_size=TEST_SIZE, stratify=strata, random_state=seed
    )
    test_set = set(test_groups)

    x_train, x_test, y_train, y_test = [], [], [], []
    for label, utterance in zip(labels, utterances):
        if utterance in test_set:
            x_test.append(utterance)
            y_test.append(label)
        else:
            x_train.append(utterance)
            y_train.append(label)
    return x_train, x_test, y_train, y_test


def get_split(split_name: str, path: Path = DEFAULT_DATA_PATH, seed: int = RANDOM_SEED) -> Split:
    """Load the data and return (x_train, x_test, y_train, y_test) for a split name."""
    labels, utterances = load_data(path)
    if split_name == "original":
        return original_split(labels, utterances, seed)
    if split_name == "grouped":
        return grouped_split(labels, utterances, seed)
    raise ValueError(f"unknown split '{split_name}', expected one of {SPLIT_NAMES}")
