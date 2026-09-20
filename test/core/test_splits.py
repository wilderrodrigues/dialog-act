from collections import Counter

import pytest

from uu import get_root
from uu.msc.ai.mair.dialog.core.splits import get_split, grouped_split, load_data, original_split

TEST_FILE = get_root() / "test" / "resources" / "test_acts.dat"
FULL_FILE = get_root() / "data" / "dialog_acts.dat"


def test_load_data_lowercases_and_keeps_all_lines() -> None:
    labels, utterances = load_data(TEST_FILE)
    assert len(labels) == len(utterances) == 10
    assert labels[0] == "inform"
    assert utterances[2] == "yes"
    assert all(u == u.lower() for u in utterances)


def test_original_split_is_stratified() -> None:
    labels, utterances = load_data(FULL_FILE)
    x_train, x_test, y_train, y_test = original_split(labels, utterances)
    assert len(x_train) + len(x_test) == len(utterances)
    assert abs(len(x_test) / len(utterances) - 0.15) < 0.01
    # every class present in the data must also be present in the test set
    assert set(y_test) == set(labels)


def test_grouped_split_has_no_shared_utterances() -> None:
    labels, utterances = load_data(FULL_FILE)
    x_train, x_test, y_train, y_test = grouped_split(labels, utterances)
    assert len(x_train) + len(x_test) == len(utterances)
    assert not set(x_train) & set(x_test)
    assert abs(len(set(x_test)) / len(set(utterances)) - 0.15) < 0.01


def test_grouped_split_keeps_duplicates_together() -> None:
    labels, utterances = load_data(FULL_FILE)
    _, x_test, _, y_test = grouped_split(labels, utterances)
    full_counts = Counter(utterances)
    test_counts = Counter(x_test)
    assert all(full_counts[u] == n for u, n in test_counts.items())


def test_get_split_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_split("random", TEST_FILE)
