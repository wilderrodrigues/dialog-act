import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402

client = TestClient(app)
PROFILE = {"age": 25, "gender": "female", "consent": True}


def start() -> dict:
    response = client.post("/api/session", json=PROFILE)
    assert response.status_code == 200
    return response.json()


def say(session: dict, message: str) -> dict:
    response = client.post("/api/chat", json={"session_id": session["session_id"], "message": message,
                                              "dialog": session["dialog"]})
    assert response.status_code == 200, response.text
    data = response.json()
    session["dialog"] = data["dialog"]
    return data


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_runs_without_torch():
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_index_is_served():
    assert "Restaurant recommender" in client.get("/").text


@pytest.mark.parametrize("profile", [{**PROFILE, "consent": False}, {**PROFILE, "age": 10},
                                     {**PROFILE, "gender": "unknown"}])
def test_invalid_profiles_are_rejected(profile):
    assert client.post("/api/session", json=profile).status_code in (400, 422)


def test_full_dialog_round_trips_state():
    session = start()
    assert session["dialog"]["state"] == "WELCOME"

    say(session, "I want a cheap restaurant in the south")
    say(session, "italian food")
    data = say(session, "no additional requirements")
    assert data["dialog"]["state"] == "SUGGEST"
    assert data["dialog"]["restaurant"] is not None

    data = say(session, "what is the phone number")
    assert data["dialog"]["state"] == "GIVE_INFO"

    data = say(session, "goodbye")
    assert data["done"] is True


def test_tampered_state_is_rejected():
    session = start()
    session["dialog"]["state"] = "NOT_A_STATE"
    response = client.post("/api/chat", json={"session_id": session["session_id"], "message": "hi",
                                              "dialog": session["dialog"]})
    assert response.status_code == 400
