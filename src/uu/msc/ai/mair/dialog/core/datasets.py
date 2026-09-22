import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torch.nn.functional as F


class DatasetFactory:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    @staticmethod
    def load_dataframe(data_path: Path, separator: str) -> pd.DataFrame:
        columns = ["act", "utterance"]
        data = []
        with data_path.open() as fd:
            full_lines = fd.read().split(sep="\n")
            for line in full_lines:
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
    def load_and_split_dataset(data_path: Path, separator: str = " ",
                               split: float = 0.15,
                               shuffle: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
        dialog_df = DatasetFactory.load_dataframe(data_path, separator)
        rows = dialog_df[dialog_df["utterance"] == "noise"].index
        dialog_df.drop(rows, inplace=True)

        targets = DatasetFactory.get_targets_map(dialog_df)

        if shuffle:
            dialog_df = dialog_df.sample(frac=1.0).reset_index(drop=True)

        train_dataset, val_dataset = train_test_split(dialog_df, test_size=split)
        return train_dataset, val_dataset, targets

    @staticmethod
    def train_tokenizer(dataset: pd.DataFrame) -> dict[str, int]:
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
        utterances = dataset.values[:,1].tolist()
        DatasetFactory.tokenizer.train_from_iterator(utterances, trainer)

        return DatasetFactory.tokenizer.get_vocab()


class DialogActsDataset(Dataset):

    def __init__(self, vocab: dict[str, int], dataframe: pd.DataFrame, targets: dict[str, int], max_tokens: int=50) -> None:
        self.vocab = vocab
        self.dataframe = dataframe
        self.targets = targets
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        act, utterance = self.dataframe.values[idx]
        tensor_act = torch.tensor(self.targets[act], dtype=torch.long)

        utterance_tokens = []
        for token in DatasetFactory.tokenizer.encode(utterance).tokens:
            utterance_tokens.append(self.vocab[token])

        tokens_idx = utterance_tokens + ([0] * (self.max_tokens - len(utterance_tokens))) if len(utterance_tokens) < self.max_tokens else utterance_tokens[:self.max_tokens]
        one_hot_encoded = F.one_hot(tensor_act, num_classes=len(self.targets)).to(torch.float)

        return torch.tensor(tokens_idx, dtype=torch.int32), one_hot_encoded
