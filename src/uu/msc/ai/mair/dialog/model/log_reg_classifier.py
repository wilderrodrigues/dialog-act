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
        self.bow_vector, self.acts = self._build_bow_vector()

    def _build_bow_vector(self) -> tuple[npt.NDArray, npt.NDArray]:
        dataframe = DatasetFactory.load_dataframe(self.dataset_path, separator=" ")
        utterances = dataframe.values[:,1]
        acts = dataframe.values[:,0]
        return self.vectorizer.fit_transform(utterances), acts

    def train(self, split: float) -> tuple[npt.NDArray, npt.NDArray]:
        utterances_train, utterances_val, acts_train, acts_val = DatasetFactory.load_and_split_vanilla_bow(bow_vector=self.bow_vector,
                                                                                                           acts=self.acts, split=split)

        self.model.fit(utterances_train, acts_train)
        acts_pred = self.model.predict(utterances_val)
        return acts_val, acts_pred
