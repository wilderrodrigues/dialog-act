import torch

from uu.msc.ai.mair.dialog.core.encoders import DistilBertTokenEncoder
from uu.msc.ai.mair.dialog.model.finetuned_classifier import FineTunedDistilBertClassifier

UTTERANCES = [
    "thank you good bye",
    "what is the phone number",
    "im looking for a cheap restaurant",
]

def test_finetuning_updates_the_pretrained_weights() -> None:
    encoder = DistilBertTokenEncoder(max_tokens=16)
    encoder.init_tokenizer(dataset=None)
    model = FineTunedDistilBertClassifier(vocabulary_size=len(encoder.get_vocabulary()), n_classes=4, max_tokens=16)

    logits = model(torch.from_numpy(encoder.encode_sentence(UTTERANCES)))
    assert logits.shape == (len(UTTERANCES), 4)
    logits.sum().backward()
    word_embeddings = model.encoder.embeddings.word_embeddings.weight
    assert word_embeddings.grad is not None
    assert word_embeddings.grad.abs().sum() > 0
