from abc import ABC, abstractmethod
import pickle
from pathlib import Path


class Predictor(ABC):

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError("This function should be implemented in a subclass.")

    @abstractmethod
    def predict_one(self, utterance: str) -> str:
        raise NotImplementedError("This function should be implemented in a subclass.")


class LogRegPredictor(Predictor):
    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self.load_model()

    def load_model(self) -> None:
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

    def predict_one(self, utterance: str) -> str:
        act = self.model.predict([utterance])
        return act[0]
