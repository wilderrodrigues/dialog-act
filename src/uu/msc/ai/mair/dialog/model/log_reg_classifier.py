from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy.typing as npt

from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory


class LogRegModel:

    def __init__(self, dataset_path: Path, num_iterations: int = 1000) -> None:
        self.dataset_path = dataset_path
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression(max_iter=num_iterations)
        self.bow_vector = None
        self.acts = None 
        self.targets = None

    def _build_bow_vector(self,data):
        return self.vectorizer.fit_transform(data)

    def train(self, split: float, seed) -> tuple[npt.NDArray, npt.NDArray]:

        utterances_train, utterances_val, acts_train, acts_val,targets = DatasetFactory.load_and_split_vanilla(data_path= self.dataset_path, separator =  " ",
                               split= split,
                               seed = seed)
        self.targets = targets
        self.bow_vector = self._build_bow_vector(utterances_train)
        vectorized_train = self.bow_vector 
        self.model.fit(vectorized_train, acts_train)
        vectorized_test = self.vectorizer.transform(utterances_val)
        acts_pred = self.model.predict(vectorized_test)
        return acts_val, acts_pred
    
    def predict(self, utterance):
        bow = self.vectorizer.transform(utterance)
        prediction = self.model.predict(bow)
        return prediction
