from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Protocol
import json
from datetime import datetime   
import pandas as pd

from uu.msc.ai.mair.recommender.core.functions import (find_restaurants, generate_response, keyword_matching,
                                                      load_restaurants, load_templates)
from uu.msc.ai.mair.recommender.core.reasoning import Candidate, apply_requirements, choose, extract_requirements


class State(IntEnum):
    """The numbered system states of the state transition diagram."""
    WELCOME = 1
    ASK_AREA = 2
    ASK_FOOD = 3
    ASK_PRICE = 4
    ASK_ADDITIONAL = 5
    SUGGEST = 6
    GIVE_INFO = 7
    TERMINATE = 8


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
    # The additional requirements of the reasoning step, e.g. {"romantic": True}; None until they have been asked.
    requirements: dict[str, bool] | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    restaurant: pd.Series | None = None
    last_utterance: str = ""


class DialogManager:
    def __init__(self, classifier: ActClassifier, restaurants_path: Path, templates_path: Path,
                 transparent: bool = True) -> None:
        """restaurants_path must point to restaurant_info_extended.csv: the reasoning rules need its extra columns.

        transparent switches the reasoning transparency on or off: show the inference chain behind a recommendation.
        """
        self.classifier = classifier
        self.restaurants = load_restaurants(restaurants_path)
        self.templates = load_templates(templates_path)
        self.transparent = transparent

    def respond(self, name: str, reasons: list[str] | None = None, **slots: str) -> str:
        return generate_response(self.templates, name, transparent=self.transparent, reasons=reasons, **slots)

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
            requirements = extract_requirements(utterance)
            if preferences or requirements or state == State.WELCOME:
                # inform (or null from Welcome): store what we learned and walk down the decision diamonds.
                context.preferences.update(preferences)
                if requirements:
                    context.requirements = {**(context.requirements or {}), **requirements}
                return self._next_question(context)
            # repeat / null / affirm without new information: stay and ask again.
            return state, self.respond(ASK_TEMPLATE[SLOT_OF_STATE[state]])

        if state == State.ASK_ADDITIONAL:
            # inform / dontcare / negate: whatever requirements were named (possibly none) are now known.
            preferences = keyword_matching(utterance, self.restaurants)
            requirements = extract_requirements(utterance)
            if act == "affirm" and not (preferences or requirements):
                # "yes" without saying which: ask again.
                return state, self.respond("askadditional")
            context.preferences.update(preferences)
            context.requirements = requirements
            return self._suggest(context)

        if state in (State.SUGGEST, State.GIVE_INFO):
            preferences = keyword_matching(utterance, self.restaurants)
            requirements = extract_requirements(utterance)
            if preferences or requirements:
                # The user changed a preference ("how about chinese food", "a romantic one"): look up again.
                context.preferences.update(preferences)
                context.requirements = {**(context.requirements or {}), **requirements}
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
        # The diamond "Additional requirements known?".
        if context.requirements is None:
            return State.ASK_ADDITIONAL, self.respond("askadditional")
        return self._suggest(context)

    def _suggest(self, context: DialogContext) -> tuple[State, str]:
        # Apply reasoning rules & filter: the lookup on the preferences, then the rules on every candidate.
        chosen, others = find_restaurants(self.restaurants, context.preferences)
        matches = [] if chosen is None else [chosen, *others]
        requirements = context.requirements or {}
        candidates, conflicts = apply_requirements(matches, requirements)
        best, context.alternatives = choose(candidates)
        context.restaurant = None if best is None else best.restaurant
        if best is not None:
            return State.SUGGEST, self.respond("recommend", reasons=best.reasons, **best.restaurant)
        if not matches:
            return State.SUGGEST, self.respond("nomatch")
        return State.SUGGEST, self._no_match_with_requirements(requirements, conflicts)

    def _no_match_with_requirements(self, requirements: dict[str, bool],
                                    conflicts: list[tuple[pd.Series, str, list[str]]]) -> str:
        wanted = " and ".join(self.respond(f"{name}{str(value).lower()}") for name, value in requirements.items())
        if not conflicts:
            return self.respond("nomatchrequirements", requirements=wanted)
        # Contradicting rules leave a property unknown; tell the user why the closest restaurant was not chosen.
        restaurant, name, reasons = conflicts[0]
        explanation = self.respond("reasonseparator").join(self.respond(reason) for reason in reasons)
        return self.respond("nomatchconflict", requirements=wanted, explanation=explanation,
                            property=self.respond(f"{name}{str(requirements[name]).lower()}"), **restaurant)

    def _alternative(self, context: DialogContext) -> tuple[State, str]:
        if not context.alternatives:
            return State.SUGGEST, self.respond("noalternatives")
        candidate = context.alternatives.pop(0)
        context.restaurant = candidate.restaurant
        return State.SUGGEST, self.respond("alternative", reasons=candidate.reasons, **candidate.restaurant)

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

    def run(self, log_dir: Path, show_acts: bool = False) -> None:
        state, context = State.WELCOME, DialogContext()
        context.last_utterance = self.respond("welcome")
        print(f"system: {context.last_utterance}")
        turns = [{"system": context.last_utterance}]
        while state != State.TERMINATE:
            try:
                utterance = input("user: ")
            except (EOFError, KeyboardInterrupt):
                break
            state, system_utterance, act = self.step(state, utterance, context)
            if show_acts:
                print(f"        [act={act}, state={state.value}. {state.name}, preferences={context.preferences}]")
            print(f"system: {system_utterance}")

            turns.append({"user": utterance, "act": act, "state": state.name, "preferences": dict(context.preferences), "system": system_utterance})
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"dialog_{datetime.now():%Y%m%d_%H%M%S}.json").open("w", encoding="utf-8") as file:
            json.dump(turns, file, indent=2)