import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

WEB_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB_DIR))

from app import app  # noqa: E402
from chatbot import storage, study  # noqa: E402
from uu.msc.ai.mair.recommender.core.storage.persistent import EXPERIMENTS_ADAPTER  # noqa: E402

client = TestClient(app)
PARTICIPANT = {"researcher": "Remy", "person": 1, "consent": True, "age": 25, "gender": "female", "ai_usage": 4}
ANSWERS = {question["id"]: 3 for question in study.QUESTIONNAIRE}


@pytest.fixture(autouse=True)
def local_database(tmp_path, monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("EXPERIMENT_DIR", str(tmp_path / "experiment"))
    return tmp_path / "experiment" / "database-remy.json"  # PARTICIPANT is run by Remy


def post(url: str, body: dict, status: int = 200) -> dict:
    response = client.post(url, json=body)
    assert response.status_code == status, response.text
    return response.json()


def converse(mode: str, messages: list[str]) -> dict:
    data = post("/api/dialog", {"mode": mode})
    dialog, transcript = data["dialog"], [{"system": data["system"]}]
    for message in messages:
        data = post("/api/chat", {"mode": mode, "message": message, "dialog": dialog})
        dialog = data["dialog"]
        transcript.append({"user": message, "act": data["act"], "restaurant": dialog["restaurant"],
                           "system": data["system"]})
    return {"done": data.get("done", False), "dialog": dialog, "transcript": transcript}


def trials_for(system: dict) -> list[dict]:
    return [{"goal_slot": goal["slot"], "goal_value": goal["value"], "started_at": "2026-10-25T12:00:00Z",
             "ended_at": "2026-10-25T12:01:30Z", "restaurant": 0, "nb_prompts": 2, "nb_resets": 1}
            for goal in system["goals"]]


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_runs_without_torch():
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_index_is_served():
    assert "Restaurant recommender study" in client.get("/").text


def test_every_person_has_a_researcher_and_both_modes():
    assert sorted(p for persons in study.RESEARCHERS.values() for p in persons) == list(study.PERSONS)
    for person in study.PERSONS:
        modes = [system["mode"] for system in study.plan(person)]
        assert modes == (["eager", "patient"] if person <= 14 else ["patient", "eager"])


def test_plans_follow_the_assignment_table():
    first, second = study.plan(1)
    assert [goal["value"] for goal in first["goals"]] == ["north", "cheap", "thai"]
    assert [goal["value"] for goal in second["goals"]] == ["east", "moderate", "international"]
    assert [goal["slot"] for goal in first["goals"]] == ["area", "pricerange", "food"]
    assert (first["mode"], first["label"]) == ("eager", "System A")
    group_b = study.plan(15)
    assert (group_b[0]["mode"], group_b[0]["label"]) == ("patient", "System A")
    assert group_b[0]["goals"] == first["goals"]


def test_goal_values_exist_in_the_restaurant_data():
    from chatbot import session
    restaurants = session.manager().restaurants
    for first, second in study.GOALS.values():
        for slot, value in zip(study.GOAL_SLOTS * 2, first + second):
            assert value in set(restaurants[slot]), (slot, value)


def test_study_overview():
    data = client.get("/api/study").json()
    assert data["researchers"]["Cai"] == list(range(8, 15))
    assert len(data["questionnaire"]) == 10
    assert data["completed"] == {} and data["storage_error"] is None


@pytest.mark.parametrize("participant", [{**PARTICIPANT, "consent": False}, {**PARTICIPANT, "age": 10},
                                         {**PARTICIPANT, "gender": "unknown"}, {**PARTICIPANT, "ai_usage": 6},
                                         {**PARTICIPANT, "person": 8}, {**PARTICIPANT, "researcher": "Nobody"}])
def test_invalid_participants_are_rejected(participant):
    assert client.post("/api/participant", json=participant).status_code in (400, 422)


def test_patient_mode_asks_every_preference():
    result = converse("patient", ["I want a cheap restaurant in the south", "italian food",
                                  "no additional requirements", "what is the phone number", "goodbye"])
    states = [turn.get("act") for turn in result["transcript"]]
    assert result["done"] is True and states[-1] == "bye"


def test_eager_mode_suggests_after_the_first_preference():
    result = converse("eager", ["I want italian food"])
    assert result["dialog"]["state"] == "SUGGEST"
    assert result["dialog"]["restaurant"] is not None


def test_tampered_state_is_rejected():
    dialog = post("/api/dialog", {"mode": "eager"})["dialog"]
    dialog["state"] = "NOT_A_STATE"
    post("/api/chat", {"mode": "eager", "message": "hi", "dialog": dialog}, status=400)


def test_results_are_stored_in_the_core_format(local_database):
    registration = post("/api/participant", PARTICIPANT)
    for system in registration["plan"]:
        post("/api/results", {**PARTICIPANT, "participant_id": registration["participant_id"],
                              "order": system["order"], "trials": trials_for(system), "answers": ANSWERS})

    records = json.loads(local_database.read_text())
    assert [(r["session"]["mode"], r["session"]["order"]) for r in records] == [("eager", 1), ("patient", 2)]
    assert {r["participant_id"] for r in records} == {registration["participant_id"]}
    first = records[0]
    assert (first["researcher"], first["person"], first["group"], first["ai_usage"]) == ("Remy", 1, "A", 4)
    trials = first["session"]["trials"]
    assert [t["goal_value"] for t in trials] == ["north", "cheap", "thai"]
    assert (trials[0]["duration"], trials[0]["nb_prompts"], trials[0]["nb_resets"]) == (90, 2, 1)
    assert trials[0]["restaurant"] and trials[0]["food_type"]
    assert [t["questionnaire"]["answers"] for t in trials] == [[], [], [[i, 3] for i in range(1, 11)]]
    assert "transcript" not in trials[0]  # conversations are not stored

    assert len(EXPERIMENTS_ADAPTER.validate_json(local_database.read_text())) == 2  # readable by the core code
    assert client.get("/api/study").json()["completed"] == {"1": [1, 2]}


def test_results_must_match_the_persons_goals():
    registration = post("/api/participant", PARTICIPANT)
    trials = trials_for(registration["plan"][1])  # the goals of the other system
    post("/api/results", {**PARTICIPANT, "participant_id": registration["participant_id"], "order": 1,
                          "trials": trials, "answers": ANSWERS}, status=400)


def test_real_participants_need_all_three_conversations():
    registration = post("/api/participant", PARTICIPANT)
    trials = trials_for(registration["plan"][0])[:1]  # only one conversation, like a test run
    post("/api/results", {**PARTICIPANT, "participant_id": registration["participant_id"], "order": 1,
                          "trials": trials, "answers": ANSWERS}, status=400)


def test_incomplete_questionnaire_is_rejected():
    registration = post("/api/participant", PARTICIPANT)
    answers = {**ANSWERS}
    answers.pop(10)
    post("/api/results", {**PARTICIPANT, "participant_id": registration["participant_id"], "order": 1,
                          "trials": trials_for(registration["plan"][0]), "answers": answers}, status=422)


def test_vercel_refuses_to_lose_results(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    registration = post("/api/participant", PARTICIPANT)
    post("/api/results", {**PARTICIPANT, "participant_id": registration["participant_id"], "order": 1,
                          "trials": trials_for(registration["plan"][0]), "answers": ANSWERS}, status=503)
    assert client.get("/api/study").json()["storage_error"]


def test_merge_keeps_every_record_once(local_database, tmp_path):
    registration = post("/api/participant", PARTICIPANT)
    system = registration["plan"][0]
    post("/api/results", {**PARTICIPANT, "participant_id": registration["participant_id"], "order": 1,
                          "trials": trials_for(system), "answers": ANSWERS})
    other = tmp_path / "other.json"
    other.write_text(local_database.read_text())
    merged = tmp_path / "merged.json"
    subprocess.run([sys.executable, str(WEB_DIR / "scripts" / "merge_results.py"), str(merged), str(local_database),
                    str(other)], check=True)
    assert len(storage.StudyData(merged).load()) == 1


def test_test_runs_are_stored_separately(local_database):
    participant = {**PARTICIPANT, "test": True}
    registration = post("/api/participant", participant)
    assert [(system["mode"], [goal["value"] for goal in system["goals"]]) for system in registration["plan"]] ==         [("eager", ["test"]), ("patient", ["test"])]  # one conversation per mode
    post("/api/results", {**participant, "participant_id": registration["participant_id"], "order": 1,
                          "trials": trials_for(registration["plan"][0]), "answers": ANSWERS})
    test_file = local_database.parent / "test-remy.json"
    assert not local_database.exists()
    assert [experiment.test for experiment in storage.StudyData(test_file).load()] == [True]
    data = client.get("/api/study").json()
    assert data["completed"] == {}  # test runs do not mark the person as done
    assert data["storage_dir"] == str(local_database.parent.resolve())
