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
    participant_id: str
    age: int
    gender: str
    sessions: list[Session]
