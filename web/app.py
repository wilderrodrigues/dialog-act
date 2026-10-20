"""Web front for the restaurant recommender: a participant form followed by the chat.

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
from pydantic import BaseModel, Field

from chatbot import session, storage

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Restaurant recommender")


class Gender(StrEnum):
    FEMALE = "female"
    MALE = "male"
    NON_BINARY = "non-binary"
    OTHER = "other"
    UNDISCLOSED = "prefer not to say"


class Profile(BaseModel):
    consent: bool
    age: int = Field(ge=16, le=120)
    gender: Gender


class ChatTurn(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=500)
    dialog: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.post("/api/session")
def start_session(profile: Profile) -> dict[str, Any]:
    if not profile.consent:
        raise HTTPException(status_code=400, detail="Consent is required to take part.")
    session_id = uuid.uuid4().hex
    storage.save_participant(session_id, {**profile.model_dump(mode="json"), "created_at": _now()})
    welcome, dialog = session.start()
    storage.save_turn(session_id, {"system": welcome, "state": dialog["state"], "time": _now()})
    return {"session_id": session_id, "system": welcome, "dialog": dialog}


@app.post("/api/chat")
def chat(turn: ChatTurn) -> dict[str, Any]:
    try:
        result = session.turn(turn.dialog, turn.message)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid dialog state: {error}") from error
    storage.save_turn(turn.session_id, {"user": turn.message, "act": result["act"], "state": result["dialog"]["state"],
                                        "preferences": result["dialog"]["preferences"], "system": result["system"],
                                        "time": _now()})
    return result


# Locally the API also serves the site; on Vercel public/ is served by the CDN and left out of the function bundle.
PUBLIC_DIR = Path(__file__).parent / "public"
if PUBLIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
