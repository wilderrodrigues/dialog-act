from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Protocol

import pandas as pd

from uu.msc.ai.mair.recommender.core.functions import (find_restaurants, generate_response, keyword_matching,
                                                      load_restaurants, load_templates)


class State(IntEnum):
    """The numbered system states of the state transition diagram."""
    WELCOME = 1
    ASK_AREA = 2
    ASK_FOOD = 3
    ASK_PRICE = 4
    SUGGEST = 5
    GIVE_INFO = 6
    TERMINATE = 7


class ActClassifier(Protocol):
    def predict_one(self, utterance: str) -> str: ...


# The order of the decision diamonds in the diagram.
SLOT_ORDER = ("area", "food", "pricerange")
ASK_STATE = {"area": State.ASK_AREA, "food": State.ASK_FOOD, "pricerange": State.ASK_PRICE}
SLOT_OF_STATE = {state: slot for slot, state in ASK_STATE.items()}
ASK_TEMPLATE = {"area": "askarea", "food": "askfoodtype", "pricerange": "askpricerange"}
INFO_TEMPLATES = {"givephone": ("phone", "number"), "giveaddress": ("address", "where"),
                  "givepostcode": ("post code", "postcode")}
# Fallbacks for acts the rule-based classifier tends to miss ("anything else" hits the 'any' inform rule first).
ALTERNATIVE_WORDS = ("else", "another", "other", "alternative")
BYE_WORDS = ("bye", "goodbye")


@dataclass
class DialogContext:
    """What the system knows so far: the user's preferences and the restaurants that match them."""
    preferences: dict[str, str | None] = field(default_factory=lambda: dict.fromkeys(SLOT_ORDER))
    alternatives: list[pd.Series] = field(default_factory=list)
    restaurant: pd.Series | None = None
    last_utterance: str = ""


class DialogManager:
    def __init__(self, classifier: ActClassifier, restaurants_path: Path, templates_path: Path) -> None:
        self.classifier = classifier
        self.restaurants = load_restaurants(restaurants_path)
        self.templates = load_templates(templates_path)

    def respond(self, name: str, **slots: str) -> str:
        return generate_response(self.templates, name, **slots)

    def transition(self, state: State, act: str, utterance: str,
                   context: DialogContext) -> tuple[State, str]:
        """The core of the dialog manager: (state, classified user utterance) -> (next state, system utterance)."""
        words = utterance.split()
        if act in ("bye", "thankyou") or any(word in BYE_WORDS for word in words):
            return State.TERMINATE, self.respond("bye")
        if act == "repeat":
            return state, context.last_utterance
        if act == "restart":
            context.__init__()
            return State.WELCOME, self.respond("welcome")

        if state in (State.WELCOME, State.ASK_AREA, State.ASK_FOOD, State.ASK_PRICE):
            preferences = keyword_matching(utterance, self.restaurants, SLOT_OF_STATE.get(state))
            if preferences or state == State.WELCOME:
                # inform (or null from Welcome): store what we learned and walk down the decision diamonds.
                context.preferences.update(preferences)
                return self._next_question(context)
            # repeat / null / affirm without new information: stay and ask again.
            return state, self.respond(ASK_TEMPLATE[SLOT_OF_STATE[state]])

        if state in (State.SUGGEST, State.GIVE_INFO):
            preferences = keyword_matching(utterance, self.restaurants)
            if preferences:
                # The user changed a preference ("how about chinese food"): look up again.
                context.preferences.update(preferences)
                return self._suggest(context)
            if act == "reqalts" or any(word in ALTERNATIVE_WORDS for word in words):
                return self._alternative(context)
            if act == "request" or self._requested_info(utterance):
                return self._give_info(utterance, context)
            return state, context.last_utterance

        return state, context.last_utterance

    def _next_question(self, context: DialogContext) -> tuple[State, str]:
        # The diamonds "Area known?", "Type food known?" and "Price range known?".
        for slot in SLOT_ORDER:
            if context.preferences[slot] is None:
                return ASK_STATE[slot], self.respond(ASK_TEMPLATE[slot])
        return self._suggest(context)

    def _suggest(self, context: DialogContext) -> tuple[State, str]:
        context.restaurant, context.alternatives = find_restaurants(self.restaurants, context.preferences)
        if context.restaurant is None:
            return State.SUGGEST, self.respond("nomatch")
        return State.SUGGEST, self.respond("recommend", **context.restaurant)

    def _alternative(self, context: DialogContext) -> tuple[State, str]:
        if not context.alternatives:
            return State.SUGGEST, self.respond("noalternatives")
        context.restaurant = context.alternatives.pop(0)
        return State.SUGGEST, self.respond("alternative", **context.restaurant)

    @staticmethod
    def _requested_info(utterance: str) -> list[str]:
        return [name for name, keywords in INFO_TEMPLATES.items() if any(k in utterance for k in keywords)]

    def _give_info(self, utterance: str, context: DialogContext) -> tuple[State, str]:
        if context.restaurant is None:
            return State.SUGGEST, context.last_utterance
        names = self._requested_info(utterance)
        if not names:
            return State.GIVE_INFO, self.respond("notunderstood")
        return State.GIVE_INFO, " ".join(self.respond(name, **context.restaurant) for name in names)

    def step(self, state: State, utterance: str, context: DialogContext) -> tuple[State, str, str]:
        utterance = utterance.strip().lower()
        act = self.classifier.predict_one(utterance)
        next_state, system_utterance = self.transition(state, act, utterance, context)
        context.last_utterance = system_utterance
        return next_state, system_utterance, act

    def run(self, show_acts: bool = False) -> None:
        state, context = State.WELCOME, DialogContext()
        context.last_utterance = self.respond("welcome")
        print(f"system: {context.last_utterance}")
        while state != State.TERMINATE:
            try:
                utterance = input("user: ")
            except (EOFError, KeyboardInterrupt):
                break
            state, system_utterance, act = self.step(state, utterance, context)
            if show_acts:
                print(f"        [act={act}, state={state.value}. {state.name}, preferences={context.preferences}]")
            print(f"system: {system_utterance}")
