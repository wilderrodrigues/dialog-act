import numpy as np
import numpy.typing as npt


class RuleBasedModel:
    KEYWORDS: dict[str, list[str]] = {
        "thankyou": ["thank", "thank you bye"],
        "affirm": ["yes"],
        "negate": ["no", "no im looking for", "no i am looking for", "no i want", "no i need", "no i would", "no in",
                   "no id like"],
        "inform": ["dont", "any", "im looking for", "i want", "i need", "no id like", "can i find", "im looking"],
        "reqalts": ["is there", "are there", "how about", "else", "about"],
        "request": ["what is", "phone number", "address", "post code"],
        "restart": ["start again", "start", "reset"],
        "reqmore": ["more"],
        "repeat": ["repeat", "back"],
        "ack": ["fine", "kay", "okay", "well"],
        "bye": ["good bye", "goodbye"],
        "deny": ["wrong"],
        "hello": ["hi", "hello", "hey"],
        "confirm": ["does it", "is it", "do they", "is this"],
        "null": [],
    }
    DEFAULT_ACT: str = "inform"

    def __init__(self) -> None:
        self.rules = [(keyword, act) for act, keywords in self.KEYWORDS.items() for keyword in keywords]

    def predict_one(self, utterance: str) -> str:
        for keyword, act in self.rules:
            if keyword in utterance:
                return act
        return self.DEFAULT_ACT

    def predict(self, utterances: npt.NDArray) -> npt.NDArray:
        return np.asarray([self.predict_one(utterance) for utterance in utterances])
