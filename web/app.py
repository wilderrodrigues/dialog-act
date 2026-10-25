"""Web front for the restaurant recommender experiment.

The researcher picks a person, the participant gives consent and fills in age, gender and AI usage, then talks to
both systems (eager and patient mode) three times each and rates each system with the questionnaire.

Run locally with ``uvicorn app:app --reload`` from this directory; on Vercel this file is the FastAPI entrypoint.
"""
import logging
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from chatbot import session, storage, study
from chatbot.session import Mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
app = FastAPI(title="Restaurant recommender")


class Gender(StrEnum):
    FEMALE = "female"
    MALE = "male"
    NON_BINARY = "non-binary"
    OTHER = "other"
    UNDISCLOSED = "prefer not to say"


class Participant(BaseModel):
    researcher: str
    person: int = Field(ge=study.PERSONS.start, lt=study.PERSONS.stop)
    consent: bool
    age: int = Field(ge=16, le=120)
    gender: Gender
    ai_usage: int = Field(ge=1, le=5)
    test: bool = False  # a trial run by the researcher: stored separately, with the goals of their first person

    @model_validator(mode="after")
    def person_belongs_to_researcher(self) -> "Participant":
        if self.person not in study.RESEARCHERS.get(self.researcher, ()):
            raise ValueError(f"Person {self.person} is not assigned to {self.researcher}.")
        return self


class DialogStart(BaseModel):
    mode: Mode


class ChatTurn(BaseModel):
    mode: Mode
    message: str = Field(min_length=1, max_length=500)
    dialog: dict[str, Any]


class TrialResult(BaseModel):
    goal_slot: str
    goal_value: str
    started_at: datetime
    ended_at: datetime
    restaurant: int | None = None
    # Only counts: the conversations themselves are not stored (nor sent).
    nb_prompts: int = Field(ge=0, le=500)
    nb_resets: int = Field(ge=0, le=500)


class SessionResult(Participant):
    participant_id: uuid.UUID
    order: int = Field(ge=1, le=2)
    # Three per system; a test run has one. save_results checks them against the person's (test) plan.
    trials: list[TrialResult] = Field(min_length=1, max_length=study.CONVERSATIONS_PER_SYSTEM)
    answers: dict[int, int]

    @model_validator(mode="after")
    def complete_questionnaire(self) -> "SessionResult":
        if set(self.answers) != study.QUESTION_IDS or not all(1 <= score <= 5 for score in self.answers.values()):
            raise ValueError("Every questionnaire item needs an answer from 1 to 5.")
        return self


@app.get("/api/study")
def study_overview() -> dict[str, Any]:
    try:
        completed, storage_error = storage.completed_sessions(), None
    except Exception as error:  # the study can still run; the researcher just does not see who is done
        logger.exception("could not list completed sessions")
        completed, storage_error = {}, str(error)
    try:
        storage_dir = str(storage.directory().resolve())
    except storage.StorageUnavailable:
        storage_dir = None
    return {**study.overview(), "completed": completed, "storage_error": storage_error, "storage_dir": storage_dir}


@app.post("/api/participant")
def register(participant: Participant) -> dict[str, Any]:
    if not participant.consent:
        raise HTTPException(status_code=400, detail="Consent is required to take part.")
    plan = study.test_plan if participant.test else study.plan
    return {"participant_id": str(uuid.uuid4()), "plan": plan(participant.person)}


@app.post("/api/dialog")
def start_dialog(start: DialogStart) -> dict[str, Any]:
    welcome, dialog = session.start()
    return {"system": welcome, "dialog": dialog}


@app.post("/api/chat")
def chat(turn: ChatTurn) -> dict[str, Any]:
    try:
        return session.turn(turn.dialog, turn.message, turn.mode)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid dialog state: {error}") from error


def _trial(result: TrialResult, answers: list[tuple[int, int]]) -> storage.StudyTrial:
    return storage.StudyTrial(
        duration=round((result.ended_at - result.started_at).total_seconds()),
        nb_resets=result.nb_resets,
        nb_prompts=result.nb_prompts,
        food_type=session.restaurant_food(result.restaurant),
        questionnaire=storage.Questionnarie(answers=answers),
        goal_slot=result.goal_slot,
        goal_value=result.goal_value,
        restaurant=session.restaurant_name(result.restaurant),
        started_at=result.started_at,
    )


@app.post("/api/results")
def save_results(result: SessionResult) -> dict[str, Any]:
    planned = (study.test_plan if result.test else study.plan)(result.person)[result.order - 1]
    if [(trial.goal_slot, trial.goal_value) for trial in result.trials] != \
            [(goal["slot"], goal["value"]) for goal in planned["goals"]]:
        raise HTTPException(status_code=400, detail="The conversations do not match this person's goals.")
    try:
        for trial in result.trials:
            session.restaurant_name(trial.restaurant)
    except KeyError as error:
        raise HTTPException(status_code=400, detail=f"Unknown restaurant: {error}") from error

    answers = sorted(result.answers.items())
    last = len(result.trials) - 1
    experiment = storage.StudyExperiment(
        participant_id=result.participant_id,
        age=result.age,
        gender=result.gender.value,
        session=storage.StudySession(
            mode=planned["mode"], label=planned["label"], order=result.order,
            trials=[_trial(trial, answers if index == last else []) for index, trial in enumerate(result.trials)],
        ),
        researcher=result.researcher,
        person=result.person,
        group=study.group_of(result.person),
        ai_usage=result.ai_usage,
        test=result.test,
        completed_at=datetime.now(timezone.utc),
    )
    try:
        storage.save_experiment(experiment)
    except Exception as error:
        logger.exception("could not save the results")
        raise HTTPException(status_code=503, detail=f"The results could not be saved: {error}") from error
    return {"saved": True}


# Locally the API also serves the site; on Vercel public/ is served by the CDN and left out of the function bundle.
PUBLIC_DIR = Path(__file__).parent / "public"
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
