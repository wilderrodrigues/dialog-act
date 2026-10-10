import torch

from uu.msc.ai.mair.dialog.model.nn_classifier import EmbeddingBagClassifier


def test_embedding_bag_classifier_returns_one_prediction_per_utterance() -> None:
    model = EmbeddingBagClassifier(vocabulary_size=10, n_classes=4, embed_len=8, padding_idx=0)
    batch = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int32)

    logits = model(batch)

    assert logits.shape == (2, 4)


def test_embedding_bag_classifier_ignores_padding() -> None:
    model = EmbeddingBagClassifier(vocabulary_size=10, n_classes=4, embed_len=8, padding_idx=0)
    tokens = torch.tensor([[1, 2]], dtype=torch.int32)
    padded_tokens = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32)

    assert torch.allclose(model(tokens), model(padded_tokens))


def test_embedding_bag_classifier_updates_non_padding_embeddings() -> None:
    model = EmbeddingBagClassifier(vocabulary_size=10, n_classes=4, embed_len=8, padding_idx=0)
    batch = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int32)

    model(batch).sum().backward()

    gradient = model.embedding_layer.weight.grad
    assert gradient is not None
    assert gradient[1:].abs().sum() > 0
    assert gradient[0].abs().sum() == 0
