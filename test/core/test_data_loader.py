import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DialogActsDataset, DatasetFactory

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

def test_data_loader() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    dialog_df = DatasetFactory.load_dataframe(dataset_file, separator=" ")

    assert dialog_df is not None
    assert dialog_df.size > 0
    assert ["act", "utterance"] in dialog_df.columns.values

    # The slice means to test the first 4 rows on column 0 (acts)
    assert set(dialog_df.values[:4,0].tolist()).issubset({"inform", "inform", "affirm", "request"})

def test_targets_map() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    dialog_df = DatasetFactory.load_dataframe(dataset_file, separator=" ")

    assert dialog_df is not None
    assert dialog_df.shape == (10, 2)

    targets_map = DatasetFactory.get_targets_map(dialog_df)
    assert targets_map is not None
    assert len(targets_map) == 5
    tgt_keys = set(targets_map.keys())
    assert tgt_keys.issubset({"inform", "affirm", "request", "reqalts", "thankyou"})

def test_load_and_split_dataset() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"

    train, val, test, targets = DatasetFactory.load_and_split_dataset(data_path=dataset_file, splits=(.5, .4))
    assert train is not None
    assert test is not None
    assert val is not None
    assert train.shape == (5, 2)
    assert val.shape == (3, 2)
    assert test.shape == (2, 2)

def test_build_vocabulary() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"

    dialog_df = DatasetFactory.load_dataframe(dataset_file, separator=" ")
    vocab = DatasetFactory.train_tokenizer(dialog_df)
    assert vocab is not None
    tokens_idx = [vocab[token] for token in DatasetFactory.tokenizer.encode("moderately priced").tokens]
    assert len(tokens_idx) == 2
    assert tokens_idx[0] == 28
    assert tokens_idx[1] == 30

def test_dataset() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    dialog_df = DatasetFactory.load_dataframe(dataset_file, separator=" ")
    assert dialog_df is not None
    assert dialog_df.shape == (10, 2)

    train, val, test, targets = DatasetFactory.load_and_split_dataset(data_path=dataset_file, splits=(.5, .4))
    vocab = DatasetFactory.train_tokenizer(dialog_df)
    assert vocab is not None

    train_dataset = DialogActsDataset(vocab=vocab, dataframe=train, targets=targets)
    val_dataset = DialogActsDataset(vocab=vocab, dataframe=val, targets=targets)
    test_dataset = DialogActsDataset(vocab=vocab, dataframe=test, targets=targets)
    assert train_dataset is not None
    assert val_dataset is not None
    assert test_dataset is not None

    assert len(train_dataset) == 5
    assert len(val_dataset) == 3
    assert len(test_dataset) == 2

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    assert train_features is not None
    assert train_labels is not None

    assert train_features.shape == (2, 50)
    assert train_labels.shape == (2, 5)

    val_features, val_labels = next(iter(val_dataloader))
    assert val_features is not None
    assert val_labels is not None
    assert val_features.shape == (2, 50)
    assert val_labels.shape == (2, 5)

    test_features, test_labels = next(iter(test_dataloader))
    assert test_features is not None
    assert test_labels is not None
    assert test_features.shape == (2, 50)
    assert test_labels.shape == (2, 5)