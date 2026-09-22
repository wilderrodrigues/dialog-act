import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

DISTILBERT_NAME = "distilbert-base-uncased"
EMBED_LEN = 768
MAX_TOKENS = 24
BATCH_SIZE = 256


class FrozenDistilBertEncoder:
    """Encodes utterances with a frozen pretrained DistilBERT model.

    The pretrained weights are never updated: the model runs in evaluation
    mode inside torch.no_grad, so it is a fixed feature extractor. Two shapes
    are offered, one per kind of classifier:

    * encode_tokens gives a (max_tokens, 768) matrix per utterance, for models
      that consume a sequence, such as the convolutional classifier;
    * encode_sentences gives a single 768 vector per utterance, for models that
      need one fixed-size feature vector, such as logistic regression.
    """

    def __init__(self, model_name: str = DISTILBERT_NAME, max_tokens: int = MAX_TOKENS,
                 batch_size: int = BATCH_SIZE) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = None

    def _load_model(self) -> None:
        """Load the tokenizer and the encoder, frozen in evaluation mode."""
        hf_logging.set_verbosity_error()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def _encode_batch(self, batch: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the hidden states of one batch and the mask of its real tokens."""
        encoded = self.tokenizer(batch, padding="max_length", truncation=True,
                                 max_length=self.max_tokens, return_tensors="pt").to(self.device)
        hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return hidden, mask

    def encode_tokens(self, utterances: list[str]) -> np.ndarray:
        """Return a (len(utterances), max_tokens, 768) array of token vectors.

        Padded positions are zeroed, so they cannot contribute to a convolution.
        """
        if self.model is None:
            self._load_model()

        # float16 halves the memory: 24k utterances at 24 tokens take 885 MB.
        embeddings = np.zeros((len(utterances), self.max_tokens, EMBED_LEN), dtype=np.float16)
        with torch.no_grad():
            for start in range(0, len(utterances), self.batch_size):
                batch = utterances[start:start + self.batch_size]
                hidden, mask = self._encode_batch(batch)
                embeddings[start:start + len(batch)] = (hidden * mask).cpu().numpy().astype(np.float16)
        return embeddings

    def encode_sentences(self, utterances: list[str]) -> np.ndarray:
        """Return a (len(utterances), 768) array with one vector per utterance.

        The token vectors are averaged over the real tokens only; the padded
        positions are excluded so that the result does not depend on how long
        the other utterances in the batch happen to be.
        """
        if self.model is None:
            self._load_model()

        embeddings = np.zeros((len(utterances), EMBED_LEN), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(utterances), self.batch_size):
                batch = utterances[start:start + self.batch_size]
                hidden, mask = self._encode_batch(batch)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                embeddings[start:start + len(batch)] = pooled.cpu().numpy()
        return embeddings
