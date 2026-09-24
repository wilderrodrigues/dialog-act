from abc import ABC, abstractmethod

import numpy as np
import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import AutoModel, AutoTokenizer, logging as hf_logging, TokenizersBackend, SentencePieceBackend

from uu.msc.ai.mair.dialog.core.runtime import select_device

DISTILBERT_NAME = "distilbert-base-uncased"
EMBED_LEN = 768
MAX_TOKENS = 50
BATCH_SIZE = 256


class Encoder(ABC):

    @abstractmethod
    def init_tokenizer(self, dataset: pd.DataFrame | None) -> None:
        raise NotImplementedError("This method must be implemented by subclasses.")

    @abstractmethod
    def get_tokenizer(self) -> None:
        raise NotImplementedError("This method must be implemented by subclasses.")

    @abstractmethod
    def get_vocabulary(self) -> None:
        raise NotImplementedError("This method must be implemented by subclasses.")

    @abstractmethod
    def encode_sentences(self, utterances: list[str]) -> np.ndarray:
        raise NotImplementedError("This method must be implemented by subclasses.")

    @abstractmethod
    def encode_sentence(self, utterance: str) -> np.ndarray:
        raise NotImplementedError("This method must be implemented by subclasses.")

class SimpleEncoder(Encoder):

    def __init__(self) -> None:
        self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def init_tokenizer(self, dataset: pd.DataFrame) -> Tokenizer:
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"])
        utterances = dataset.values[:,1].tolist()
        self.tokenizer.train_from_iterator(utterances, trainer)

        return self.tokenizer

    def get_tokenizer(self) -> Tokenizer:
        return self.tokenizer

    def encode_sentences(self, utterances: list[str]) -> list[int]:
        return self.tokenizer.encode(utterances, is_pretokenized=True).ids

    def encode_sentence(self, utterance: str) -> list[int]:
        return self.tokenizer.encode(utterance).ids

    def get_vocabulary(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

class FrozenDistilBertEncoder(Encoder):

    def __init__(self, model_name: str = DISTILBERT_NAME, max_tokens: int = MAX_TOKENS,
                 batch_size: int = BATCH_SIZE) -> None:
        hf_logging.set_verbosity_error()

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = select_device()
        self.tokenizer = None
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def _encode_batch(self, batch: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(batch, padding="max_length", truncation=True,
                                 max_length=self.max_tokens, return_tensors="pt").to(self.device)
        hidden_state = self.model(**encoded).last_hidden_state
        return hidden_state * encoded["attention_mask"].unsqueeze(-1).to(hidden_state.dtype)

    def init_tokenizer(self, dataset: pd.DataFrame | None) -> TokenizersBackend | SentencePieceBackend:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def get_tokenizer(self) -> TokenizersBackend | SentencePieceBackend:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call 'init_tokenizer' first.")
        return self.tokenizer

    def get_vocabulary(self) -> dict[str, int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call 'init_tokenizer' first.")
        return self.tokenizer.get_vocab()

    def encode_sentence(self, utterance: str | list[str]) -> np.ndarray:
        is_single = isinstance(utterance, str)
        utterances = [utterance] if is_single else list(utterance)
        embeddings = np.zeros((len(utterances), self.max_tokens, EMBED_LEN), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(utterances), self.batch_size):
                batch = utterances[start:start + self.batch_size]
                masked = self._encode_batch(batch)
                embeddings[start:start + len(batch)] = masked.detach().cpu().numpy()
        return embeddings[0] if is_single else embeddings

    def encode_sentences(self, utterances: list[str]) -> np.ndarray:
        embeddings = np.zeros((len(utterances), EMBED_LEN), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(utterances), self.batch_size):
                batch = list(utterances[start:start + self.batch_size])
                pooled = self._encode_batch(batch)[:, 0]
                embeddings[start:start + len(batch)] = pooled.detach().cpu().numpy()
        return embeddings
