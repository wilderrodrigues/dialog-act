"""Runs the existing DialogManager one turn per HTTP request.

Vercel Functions keep no memory between requests, so the dialog state and context travel with every request as
JSON. Restaurants are stored by their row label in ``restaurant_info_extended.csv`` and looked up again on load.
"""
from functools import lru_cache
from typing import Any

from chatbot import core_loader

core_loader.install()

from uu.msc.ai.mair.recommender.core.dialog_manager import DialogContext, DialogManager, Fallback, Mode, State  # noqa: E402
from uu.msc.ai.mair.recommender.core.inference_engine import LogRegPredictor  # noqa: E402
from uu.msc.ai.mair.recommender.core.reasoning import Candidate  # noqa: E402


@lru_cache(maxsize=1)
def classifier() -> LogRegPredictor:
    with core_loader.posix_paths_on_windows():
        return LogRegPredictor(core_loader.MODEL_PATH)


@lru_cache(maxsize=len(Mode))
def manager(mode: Mode = Mode.PATIENT) -> DialogManager:
    # The web app stores the results itself (chatbot.storage), so the manager gets no experiment storage.
    return DialogManager(classifier=classifier(), restaurants_path=core_loader.RESTAURANTS_PATH,
                         templates_path=core_loader.TEMPLATES_PATH, experiment_storage=None,
                         transparent=True, fallback=Fallback.LEVENSHTEIN, mode=mode)


def dump(state: State, context: DialogContext) -> dict[str, Any]:
    return {
        "state": state.name,
        "preferences": dict(context.preferences),
        "requirements": context.requirements,
        "alternatives": [{"restaurant": int(c.restaurant.name), "reasons": c.reasons} for c in context.alternatives],
        "restaurant": None if context.restaurant is None else int(context.restaurant.name),
        "last_utterance": context.last_utterance,
    }


def load(data: dict[str, Any]) -> tuple[State, DialogContext]:
    restaurants = manager().restaurants
    context = DialogContext()
    context.preferences.update({slot: data["preferences"].get(slot) for slot in context.preferences})
    context.requirements = data["requirements"]
    context.alternatives = [Candidate(restaurants.loc[a["restaurant"]], list(a["reasons"]))
                            for a in data["alternatives"]]
    context.restaurant = None if data["restaurant"] is None else restaurants.loc[data["restaurant"]]
    context.last_utterance = data["last_utterance"]
    return State[data["state"]], context


def start() -> tuple[str, dict[str, Any]]:
    context = DialogContext()
    context.last_utterance = manager().respond("welcome")
    return context.last_utterance, dump(State.WELCOME, context)


def turn(data: dict[str, Any], utterance: str, mode: Mode = Mode.PATIENT) -> dict[str, Any]:
    state, context = load(data)
    if state == State.TERMINATE:
        raise ValueError("The dialog has already ended.")
    next_state, system_utterance, act = manager(mode).step(state, utterance, context)
    return {"system": system_utterance, "act": act, "done": next_state == State.TERMINATE,
            "dialog": dump(next_state, context)}


def restaurant_name(row: int | None) -> str | None:
    return None if row is None else str(manager().restaurants.loc[row, "restaurantname"])


def restaurant_food(row: int | None) -> str:
    return "" if row is None else str(manager().restaurants.loc[row, "food"])
