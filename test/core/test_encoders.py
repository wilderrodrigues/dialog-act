import numpy as np
import pytest

from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory
from uu.msc.ai.mair.dialog.core.encoders import EMBED_LEN, FrozenDistilBertEncoder, SimpleEncoder

UTTERANCES = [
    "thank you good bye",
    "what is the phone number",
    "im looking for a cheap restaurant",
]


@pytest.fixture(scope="module")
def simple_encoder() -> SimpleEncoder:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    dialog_df = DatasetFactory.load_dataframe(dataset_file, separator=" ")
    simple_encoder = SimpleEncoder()
    simple_encoder.init_tokenizer(dialog_df)
    return simple_encoder


def test_simple_vocabulary(simple_encoder: SimpleEncoder) -> None:
    vocab = simple_encoder.get_vocabulary()
    assert vocab is not None


def test_simple_encode_sentence(simple_encoder: SimpleEncoder) -> None:
    tokens_idx = simple_encoder.encode_sentence("moderately priced")
    assert len(tokens_idx) == 2
    assert tokens_idx[0] == 35
    assert tokens_idx[1] == 38


def test_simple_encode_sentences(simple_encoder: SimpleEncoder) -> None:
    tokens_idx = simple_encoder.encode_sentences(["moderately priced", "i am looking for a cheap restaurant"])
    assert len(tokens_idx) == 9
    assert tokens_idx == [35, 38, 10, 1, 34, 27, 20, 1, 13]


@pytest.fixture(scope="module")
def bert_encoder() -> FrozenDistilBertEncoder:
    encoder = FrozenDistilBertEncoder(max_tokens=16, batch_size=2)
    encoder.init_tokenizer(dataset=None)
    return encoder


def test_encode_sentences_has_one_vector_per_utterance(bert_encoder: FrozenDistilBertEncoder) -> None:
    embeddings = bert_encoder.encode_sentences(UTTERANCES)
    assert embeddings.shape == (len(UTTERANCES), EMBED_LEN)
    assert np.isfinite(embeddings).all()


def test_padding_does_not_change_the_result(bert_encoder: FrozenDistilBertEncoder) -> None:
    alone = bert_encoder.encode_sentences(["thank you"])
    with_others = bert_encoder.encode_sentences(["thank you", "im looking for an expensive restaurant "
                                                          "in the north part of town"])
    assert np.allclose(alone[0], with_others[0], atol=1e-4)


def test_encoder_is_frozen(bert_encoder: FrozenDistilBertEncoder) -> None:
    assert not bert_encoder.model.training
    assert all(not parameter.requires_grad or parameter.grad is None
               for parameter in bert_encoder.model.parameters())


def test_encoding_is_deterministic(bert_encoder: FrozenDistilBertEncoder) -> None:
    first = bert_encoder.encode_sentences(UTTERANCES)
    second = bert_encoder.encode_sentences(UTTERANCES)
    assert np.array_equal(first, second)

def test_bert_vocabulary(bert_encoder: FrozenDistilBertEncoder) -> None:
    vocab = bert_encoder.get_vocabulary()
    assert len(vocab) == 30522

    tokens = [vocab[word] for word in "moderately priced".split(sep=" ")]
    assert tokens == [17844, 21125]
