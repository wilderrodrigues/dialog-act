from typing import Iterator

import torch
from torch import nn
from transformers import AutoModel, logging as hf_logging

from uu.msc.ai.mair.dialog.core.encoders import DISTILBERT_NAME, EMBED_LEN, MAX_TOKENS
from uu.msc.ai.mair.dialog.model.nn_classifier import Conv1DClassifier


class FineTunedDistilBertClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, n_classes: int, max_tokens: int=MAX_TOKENS,
                 model_name: str=DISTILBERT_NAME):
        super().__init__()
        hf_logging.set_verbosity_error()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = Conv1DClassifier(vocabulary_size=vocabulary_size, n_classes=n_classes,
                                     max_tokens=max_tokens, embed_len=EMBED_LEN)
        self.head.embedding_layer = nn.Identity()

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        return self.head.parameters()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        input_ids, attention_mask = batch[:, 0], batch[:, 1]
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = hidden_state * attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return self.head(x)
