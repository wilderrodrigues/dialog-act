from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy.typing as npt

from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory
from uu.msc.ai.mair.dialog.core.encoders import FrozenDistilBertEncoder


class LogRegModel:

    def __init__(self, dataset_path: Path, num_iterations: int = 1000, encoder: str = "bow") -> None:
        self.dataset_path = dataset_path
        self.encoder = encoder
        if (encoder == "bert"):
            self.vectorizer = FrozenDistilBertEncoder()
            self.vectorizer.init_tokenizer(None)
        else:
            self.vectorizer = CountVectorizer()
        
        self.model = LogisticRegression(max_iter=num_iterations)
        self.bow_vector = None
        self.acts = None 
        self.targets = None

    def _vectorize(self, utterances: npt.NDArray, fit: bool = False)->npt.NDArray:
        if isinstance(self.vectorizer, FrozenDistilBertEncoder):
            return self.vectorizer.encode_sentences(list(utterances))
        return self.vectorizer.fit_transform(utterances) if fit else self.vectorizer.transform(utterances)
    
    def _build_bow_vector(self, data: npt.NDArray) -> npt.NDArray:
        return self.vectorizer.fit_transform(data)

    def train(self, split: float, seed: int, split_strategy: str = "vanilla") -> tuple[npt.NDArray, npt.NDArray]:

        load_and_split = (DatasetFactory.load_and_split_vanilla if split_strategy == "vanilla" else DatasetFactory.load_and_split_grouped)
        utterances_train, utterances_val, acts_train, acts_val, targets = load_and_split(
        data_path=self.dataset_path, separator=" ", split=split, seed=seed)

        self.targets = targets
        vectorized_train = self._vectorize(utterances_train, fit = True) 
        self.model.fit(vectorized_train, acts_train)
        vectorized_test = self._vectorize(utterances_val)
        acts_pred = self.model.predict(vectorized_test)
        return acts_val, acts_pred
    
    def predict(self, utterance: list[str]) -> npt.NDArray:
        bow = self._vectorize(utterance)
        prediction = self.model.predict(bow)
        return prediction
