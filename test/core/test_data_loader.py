import random
import torch
import numpy as np
import numpy.typing as npt
from numba.np.arraymath import np_convolve
from torch.utils.data import DataLoader

from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DialogActsDataset, DatasetFactory
from uu.msc.ai.mair.dialog.core.encoders import SimpleEncoder

random.seed(DatasetFactory.SEED)
torch.manual_seed(DatasetFactory.SEED)
np.random.seed(DatasetFactory.SEED)

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
    assert dialog_df.shape == (18, 2)

    targets_map = DatasetFactory.get_targets_map(dialog_df)
    assert targets_map is not None
    assert len(targets_map) == 5
    tgt_keys = set(targets_map.keys())
    assert tgt_keys.issubset({"inform", "affirm", "request", "reqalts", "thankyou"})

def test_load_and_split_vanilla() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"

    utterances_train, utterances_val, _, _, targets = (
        DatasetFactory.load_and_split_vanilla(data_path=dataset_file, split=.5))
    assert utterances_train is not None
    assert utterances_val is not None
    assert utterances_train.shape == (9,)
    assert utterances_val.shape == (9,)

def test_load_and_split_grouped() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    # dataset_file = get_root() / "data" / "dialog_acts.dat"

    utterances_train, utterances_val, _, _, targets = (
        DatasetFactory.load_and_split_grouped(data_path=dataset_file, split=.5))
    assert utterances_train is not None
    assert utterances_val is not None
    assert utterances_train.shape == (10,)
    assert utterances_val.shape == (8,)


def test_dataset() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    dialog_df = DatasetFactory.load_dataframe(dataset_file, separator=" ")
    assert dialog_df is not None
    assert dialog_df.shape == (18, 2)

    utterances_train, utterances_val, acts_train, acts_val, targets = (
        DatasetFactory.load_and_split_vanilla(data_path=dataset_file, split=.5))

    encoder = SimpleEncoder()
    encoder.init_tokenizer(dialog_df)
    assert encoder.get_tokenizer() is not None
    assert encoder.get_vocabulary() is not None

    train_dataset = DialogActsDataset(encoder=encoder, utterances=utterances_train, acts=acts_train, targets=targets)
    val_dataset = DialogActsDataset(encoder=encoder, utterances=utterances_val, acts=acts_val, targets=targets)
    assert train_dataset is not None
    assert val_dataset is not None

    assert len(train_dataset) == 9
    assert len(val_dataset) == 9

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    assert train_features is not None
    assert train_labels is not None
    assert train_features.shape == (2, 50)
    assert train_labels.shape == (2, 5)

    test_features, test_labels = next(iter(test_dataloader))
    assert test_features is not None
    assert test_labels is not None
    assert test_features.shape == (2, 50)
    assert test_labels.shape == (2, 5)

def convolve(features: npt.NDArray[np.float32], kernel: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    padding = 1
    stride = 1
    height, width, channels = features.shape
    kernel_height, kernel_width, _ = kernel.shape
    out_h = int((height - kernel_height) + (2 * padding) / stride) + 1
    out_w = int((width - kernel_width) + (2 * padding) / stride) + 1
    features = np.pad(features, ((padding, padding), (padding, padding), (0, 0)), 'constant')

    i0 = np.repeat(np.arange(kernel_height), kernel_height)
    i1 = np.repeat(np.arange(height), height)
    j0 = np.tile(np.arange(kernel_width), kernel_height)
    j1 = np.tile(np.arange(height), width)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), kernel_height * kernel_width).reshape(-1, 1)

    select_img = features[i, j, :].squeeze()  # receptive feild pixels are selected based on the index***[9, 100, 3] reshaped to [9, 100, 3]
    weights = kernel.reshape(kernel_height * kernel_width, -1)  # weights reshaped to [9, 3]
    convolved = weights.transpose() @ np.permute_dims(select_img, [0, 2, 1])  # convolution operation [3, 9] * [9, 3, 100] ----> [9, 9, 100]
    convolved = convolved.reshape(10, 10, 3)  # reshaped in image dimension [10, 10, 3]
    return convolved

def test_convolve() -> None:
    random.seed(42)
    synth_img = np.random.rand(10, 10, 3).astype(np.float32)
    assert synth_img.shape == (10, 10, 3)

    sobel_filter = np.array([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]).astype(np.float32)
    sobel_filter = np.repeat(sobel_filter[:, :, np.newaxis], synth_img.shape[2], axis=2)
    conv_result = convolve(synth_img, sobel_filter)
    assert conv_result.shape == (10, 10, 3)

    np_result = np.convolve(synth_img, sobel_filter, mode='same')
    assert np_result.shape == (10, 10, 3)
    assert np.allclose(conv_result, np_result)
