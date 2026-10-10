from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy.typing as npt

from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory


class LogRegModel:
    def __init__(self, num_iterations: int = 1000,use_bow: bool = True, seed = int) -> None:
        self.use_bow = use_bow
        self.vectorizer = CountVectorizer() if use_bow else None
        self.model = LogisticRegression(max_iter=num_iterations, random_state=seed)
        self.targets = None
 
    def fit(self, train_x, train_acts, targets: dict[str, int]) -> None:
        self.targets = targets
        if self.vectorizer is not None:
            train_x = self.vectorizer.fit_transform(train_x)
        self.model.fit(train_x, train_acts)

 
    def predict(self, x):
        if self.vectorizer is not None:
            x = self.vectorizer.transform(x)
        return self.model.predict(x)
