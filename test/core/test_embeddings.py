import numpy as np
import pytest

from uu.msc.ai.mair.dialog.core.embeddings import EMBED_LEN, FrozenDistilBertEncoder

UTTERANCES = [
    "thank you good bye",
    "what is the phone number",
    "im looking for a cheap restaurant",
]


@pytest.fixture(scope="module")
def encoder() -> FrozenDistilBertEncoder:
    return FrozenDistilBertEncoder(max_tokens=16, batch_size=2)


def test_encode_tokens_has_one_matrix_per_utterance(encoder: FrozenDistilBertEncoder) -> None:
    embeddings = encoder.encode_tokens(UTTERANCES)
    assert embeddings.shape == (len(UTTERANCES), encoder.max_tokens, EMBED_LEN)
    assert np.isfinite(embeddings).all()


def test_encode_sentences_has_one_vector_per_utterance(encoder: FrozenDistilBertEncoder) -> None:
    embeddings = encoder.encode_sentences(UTTERANCES)
    assert embeddings.shape == (len(UTTERANCES), EMBED_LEN)
    assert np.isfinite(embeddings).all()


def test_padded_positions_are_zero(encoder: FrozenDistilBertEncoder) -> None:
    embeddings = encoder.encode_tokens(["thank you"])
    n_real_tokens = int(sum(encoder.tokenizer("thank you")["attention_mask"]))
    assert np.any(embeddings[0, :n_real_tokens] != 0)
    assert np.all(embeddings[0, n_real_tokens:] == 0)


def test_sentence_vector_is_the_mean_of_the_real_tokens(encoder: FrozenDistilBertEncoder) -> None:
    tokens = encoder.encode_tokens(UTTERANCES).astype(np.float32)
    sentences = encoder.encode_sentences(UTTERANCES)
    for idx, utterance in enumerate(UTTERANCES):
        n_real_tokens = int(sum(encoder.tokenizer(utterance)["attention_mask"]))
        expected = tokens[idx, :n_real_tokens].mean(axis=0)
        assert np.allclose(expected, sentences[idx], atol=1e-2)


def test_padding_does_not_change_the_result(encoder: FrozenDistilBertEncoder) -> None:
    """A short utterance keeps its vector when encoded next to a longer one."""
    alone = encoder.encode_sentences(["thank you"])
    with_others = encoder.encode_sentences(["thank you", "im looking for an expensive restaurant "
                                                          "in the north part of town"])
    assert np.allclose(alone[0], with_others[0], atol=1e-4)


def test_encoder_is_frozen(encoder: FrozenDistilBertEncoder) -> None:
    """The pretrained model stays in evaluation mode and is never trained."""
    encoder.encode_sentences(UTTERANCES)
    assert not encoder.model.training
    assert all(not parameter.requires_grad or parameter.grad is None
               for parameter in encoder.model.parameters())


def test_encoding_is_deterministic(encoder: FrozenDistilBertEncoder) -> None:
    first = encoder.encode_sentences(UTTERANCES)
    second = encoder.encode_sentences(UTTERANCES)
    assert np.array_equal(first, second)
