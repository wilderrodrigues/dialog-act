import uuid
import pydantic
from pydantic.dataclasses import dataclass



@dataclass(frozen=True)
class Questionnarie:
    answers: list[tuple[int, int]]


@dataclass(frozen=True)
class Trial:
    duration: int
    nb_resets: int
    nb_prompts: int
    food_type: str
    questionnaire: Questionnarie


@dataclass(frozen=True)
class Session:
    mode: str
    trials: list[Trial]


@dataclass(frozen=True)
class Experiment:
    age: int
    gender: str
    session: Session
    participant_id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
