import numpy.typing as npt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torch.nn.functional as F


class DatasetFactory:

    VAL_SPLIT: int = 0.15
    SEED: int = 42

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    @staticmethod
    def load_dataframe(data_path: Path, separator: str) -> pd.DataFrame:
        columns = ["act", "utterance"]
        data = []
        with data_path.open() as fd:
            full_lines = fd.read().split(sep="\n")
            for line in full_lines:
                if line is None or line.strip() == "":
                    continue
                tokens = line.split(separator)
                act = tokens[0]
                utterance = " ".join(tokens[1:])
                data.append([act, utterance])

        dialog_df = pd.DataFrame(data, columns=columns)
        return dialog_df

    @staticmethod
    def get_targets_map(dialog_df: pd.DataFrame) -> dict[str, int]:
        targets_map = {act: idx for idx, act in enumerate(dialog_df.act.unique())}
        return targets_map

    @staticmethod
    def load_and_split_vanilla(data_path: Path, separator: str = " ",
                               split: float = VAL_SPLIT,
                               shuffle: bool = True,
                               seed: int = SEED) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, dict[str, int]]:
        dialog_df = DatasetFactory.load_dataframe(data_path, separator)
        rows = dialog_df[dialog_df["utterance"] == "noise"].index
        dialog_df.drop(rows, inplace=True)

        targets = DatasetFactory.get_targets_map(dialog_df)

        utterances = dialog_df.values[:,1]
        acts = dialog_df.values[:,0]

        utterances_train, utterances_val, acts_train, acts_val = train_test_split(utterances, acts,
                                                                                  stratify=acts, test_size=split,
                                                                                  random_state=seed,
                                                                                  shuffle=shuffle)
        return utterances_train, utterances_val, acts_train, acts_val, targets

    @staticmethod
    def load_and_split_grouped(data_path: Path, separator: str = " ",
                               split: float = VAL_SPLIT,
                               shuffle: bool = True,
                               seed: int = SEED) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, dict[str, int]]:
        dialog_df = DatasetFactory.load_dataframe(data_path, separator)
        rows = dialog_df[dialog_df["utterance"] == "noise"].index
        dialog_df.drop(rows, inplace=True)

        targets = DatasetFactory.get_targets_map(dialog_df)
        utterances = dialog_df.values[:, 1]
        acts = dialog_df.values[:, 0]

        utterances_train = None
        utterances_val = None
        acts_train = None
        acts_val = None

        grouped_dataset = StratifiedGroupKFold(n_splits=2, random_state=seed, shuffle=shuffle)
        for idx, (train_index, val_index) in enumerate(grouped_dataset.split(utterances, acts, utterances)):
            utterances_train = utterances[train_index]
            utterances_val = utterances[val_index]
            acts_train = acts[train_index]
            acts_val = acts[val_index]
            # TODO [Wilder]:
            # We are deliberately not using the second fold here. The idea is that fold-1 has the duplicates, which
            # are not included in fold-2. Then, the second fold will be the other way around. This is to make
            # sure that the model is trained with both folds, but without leaking the data from train to validation.
            # We will discuss this with Professor Roxana.
            break

        return utterances_train, utterances_val, acts_train, acts_val, targets

    @staticmethod
    def train_tokenizer(dataset: pd.DataFrame) -> dict[str, int]:
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
        utterances = dataset.values[:,1].tolist()
        DatasetFactory.tokenizer.train_from_iterator(utterances, trainer)

        return DatasetFactory.tokenizer.get_vocab()


class DialogActsDataset(Dataset):

    def __init__(self, vocab: dict[str, int], utterances: npt.NDArray, acts: npt.NDArray, targets: dict[str, int],
                 max_tokens: int=50) -> None:
        self.vocab = vocab
        self.utterances = utterances
        self.acts = acts
        self.targets = targets
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        act, utterance = self.acts[idx], self.utterances[idx]
        tensor_act = torch.tensor(self.targets[act], dtype=torch.long)

        utterance_tokens = []
        for token in DatasetFactory.tokenizer.encode(utterance).tokens:
            utterance_tokens.append(self.vocab[token])

        tokens_idx = utterance_tokens + ([0] * (self.max_tokens - len(utterance_tokens))) if len(utterance_tokens) < self.max_tokens else utterance_tokens[:self.max_tokens]
        one_hot_encoded = F.one_hot(tensor_act, num_classes=len(self.targets)).to(torch.float)

        return torch.tensor(tokens_idx, dtype=torch.int32), one_hot_encoded
