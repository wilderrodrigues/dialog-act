from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
import json
from datetime import datetime   
import pandas as pd
import logging
import re

from uu.msc.ai.mair.recommender.core.functions import (find_restaurants, generate_response, keyword_matching,
                                                      load_restaurants, load_templates, levenshtein_distance, semantic_similarity)
from uu.msc.ai.mair.recommender.core.inference_engine import Predictor
from uu.msc.ai.mair.recommender.core.reasoning import Candidate, apply_requirements, choose, extract_requirements
from uu.msc.ai.mair.dialog.core.encoders import FrozenDistilBertEncoder

class State(IntEnum):
    WELCOME = 1
    ASK_AREA = 2
    ASK_FOOD = 3
    ASK_PRICE = 4
    ASK_ADDITIONAL = 5
    SUGGEST = 6
    GIVE_INFO = 7
    TERMINATE = 8


class DialogAct(StrEnum):
    INFORM = "inform"
    AFFIRM = "affirm"
    NEGATE = "negate"
    REQUEST = "request"
    REQALTS = "reqalts"
    THANKYOU = "thankyou"
    NULL = "null"
    BYE = "bye"
    REPEAT = "repeat"
    RESTART = "restart"
    ACK = "ack"
    DENY = "deny"
    CONFIRM = "confirm"
    HELLO = "hello"
    REQMORE = "reqmore"


class Fallback(StrEnum):
    NONE = "none"
    LEVENSHTEIN = "levenshtein"
    BERT = "bert"

class Mode(StrEnum):
    PATIENT = "patient"
    EAGER = "eager"
CLOSING_WORDS = {"thanks", "thank", "you", "bye", "goodbye", "okay", "ok", "great", "cheers"}    
SLOT_ORDER = ("area", "food", "pricerange")
ASK_STATE = {"area": State.ASK_AREA, "food": State.ASK_FOOD, "pricerange": State.ASK_PRICE}
SLOT_OF_STATE = {state: slot for slot, state in ASK_STATE.items()}
ASK_TEMPLATE = {"area": "askarea", "food": "askfoodtype", "pricerange": "askpricerange"}
INFO_TEMPLATES = {"givephone": ("phone", "number"), "giveaddress": ("address", "where"),
                  "givepostcode": ("post code", "postcode")}

@dataclass
class DialogContext:
    preferences: dict[str, str | None] = field(default_factory=lambda: dict.fromkeys(SLOT_ORDER))
    requirements: dict[str, bool] | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    restaurant: pd.Series | None = None
    last_utterance: str = ""

    def reset(self) -> None:
        self.preferences = dict.fromkeys(SLOT_ORDER)
        self.requirements = None
        self.alternatives.clear()
        self.restaurant = None
        self.last_utterance = ""


class DialogManager:
    def __init__(self, classifier: Predictor, restaurants_path: Path, templates_path: Path,
                 transparent: bool = True, fallback: Fallback = Fallback.LEVENSHTEIN, mode: Mode = Mode.PATIENT) -> None:
        self.classifier = classifier
        self.restaurants = load_restaurants(restaurants_path)
        self.templates = load_templates(templates_path)
        self.transparent = transparent
        self.fallback = fallback
        self.mode = mode
        self.encoder = None
        if fallback == Fallback.BERT:
            self.encoder = FrozenDistilBertEncoder()
            self.encoder.init_tokenizer(None)

    def _extract(self, utterance: str, asked_slot: str | None = None) -> dict[str, str]:
        preferences = keyword_matching(utterance, self.restaurants, asked_slot)
        if self.fallback == Fallback.LEVENSHTEIN:
            closest = levenshtein_distance(utterance, self.restaurants, preferences)
        elif self.fallback == Fallback.BERT:
            closest = semantic_similarity(utterance, self.restaurants, preferences, self.encoder)
        else:
            closest = {}
        return preferences | {slot: value for slot, (value, _) in closest.items()}

    def respond(self, name: str, reasons: list[str] | None = None, **slots: str) -> str:
        return generate_response(self.templates, name, transparent=self.transparent, reasons=reasons, **slots)

    def transition(self, state: State, act: str, utterance: str,
                   context: DialogContext) -> tuple[State, str]:
        try:
            dialog_act = DialogAct(act)
        except ValueError:
            return self._recover(state)

        if dialog_act in (DialogAct.BYE, DialogAct.THANKYOU):
            return State.TERMINATE, self.respond("bye")
        if dialog_act == DialogAct.REPEAT:
            return state, context.last_utterance
        if dialog_act == DialogAct.RESTART:
            context.reset()
            return State.WELCOME, self.respond("welcome")

        handlers = {
            State.WELCOME: self._handle_welcome,
            State.ASK_AREA: self._handle_slot_question,
            State.ASK_FOOD: self._handle_slot_question,
            State.ASK_PRICE: self._handle_slot_question,
            State.ASK_ADDITIONAL: self._handle_additional_requirements,
            State.SUGGEST: self._handle_recommendation_state,
            State.GIVE_INFO: self._handle_recommendation_state,
        }
        handler = handlers.get(state)
        if handler is None:
            return state, context.last_utterance
        return handler(state, dialog_act, utterance, context)

    def _handle_welcome(self, state: State, act: DialogAct, utterance: str,
                        context: DialogContext) -> tuple[State, str]:
        if act == DialogAct.NULL:
            return State.ASK_AREA, self.respond(ASK_TEMPLATE["area"])
        if act != DialogAct.INFORM:
            return self._recover(state)
        self._store_early_information(utterance, context)
        return self._next_question(context)

    def _handle_slot_question(self, state: State, act: DialogAct, utterance: str,
                              context: DialogContext) -> tuple[State, str]:
        if act in (DialogAct.AFFIRM, DialogAct.NULL):
            return state, self.respond(ASK_TEMPLATE[SLOT_OF_STATE[state]])
        if act != DialogAct.INFORM:
            return self._recover(state)

        preferences, requirements = self._extract_information(utterance, SLOT_OF_STATE[state])
        if not preferences and not requirements:
            return self._recover(state)
        context.preferences.update(preferences)
        if requirements:
            context.requirements = {**(context.requirements or {}), **requirements}
        return self._next_question(context)

    def _handle_additional_requirements(self, state: State, act: DialogAct, utterance: str,
                                        context: DialogContext) -> tuple[State, str]:
        if act == DialogAct.NEGATE:
            context.requirements = {}
            return self._suggest(context)
        if act not in (DialogAct.INFORM, DialogAct.NULL):
            return self._recover(state)

        preferences, requirements = self._extract_information(utterance)
        if requirements:
            context.requirements = {**(context.requirements or {}), **requirements}
        elif self._declines_requirements(utterance):
            context.requirements = {}
        else:
            return self._recover(state)
        context.preferences.update(preferences)
        return self._suggest(context)

    def _handle_recommendation_state(self, state: State, act: DialogAct, utterance: str,
                                     context: DialogContext) -> tuple[State, str]:
        if act == DialogAct.REQALTS:
            preferences = keyword_matching(utterance, self.restaurants)
            if preferences:
                context.preferences.update(preferences)
                context.alternatives.clear()
                return self._suggest(context)
            return self._alternative(context)
        if act == DialogAct.REQUEST:
            return self._give_info(utterance, context)
        if act != DialogAct.INFORM:
            return self._recover(state)

        preferences, requirements = self._extract_information(utterance)
        if not preferences and not requirements:
            return self._recover(state)
        context.preferences.update(preferences)
        if requirements:
            context.requirements = {**(context.requirements or {}), **requirements}
        context.alternatives.clear()
        return self._suggest(context)

    def _extract_information(self, utterance: str, asked_slot: str | None = None
                             ) -> tuple[dict[str, str], dict[str, bool]]:
        preferences = self._extract(utterance, asked_slot)
        requirements = extract_requirements(utterance)
        return preferences, requirements

    def _store_early_information(self, utterance: str, context: DialogContext) -> None:
        preferences, requirements = self._extract_information(utterance)
        context.preferences.update(preferences)
        if requirements:
            context.requirements = {**(context.requirements or {}), **requirements}

    @staticmethod
    def _closing_act(act: str, utterance: str) -> str:
        words = set(utterance.split())
        if words <= CLOSING_WORDS:
            if words & {"bye", "goodbye"}:
                return DialogAct.BYE
            if words & {"thanks", "thank"}:
                return DialogAct.THANKYOU
        return act

    @staticmethod
    def _declines_requirements(utterance: str) -> bool:
        words = utterance.replace("'", "").lower().split()
        phrases = ("no requirements", "no additional requirements", "dont care", "doesnt matter",
                   "does not matter", "whatever", "nothing else")
        return any(phrase in " ".join(words) for phrase in phrases) or "any" in words or "none" in words

    def _recover(self, state: State) -> tuple[State, str]:
        response = self.respond("notunderstood")
        if state in SLOT_OF_STATE:
            response = f"{response} {self.respond(ASK_TEMPLATE[SLOT_OF_STATE[state]])}"
        elif state == State.ASK_ADDITIONAL:
            response = f"{response} {self.respond('askadditional')}"
        elif state == State.WELCOME:
            response = f"{response} {self.respond('welcome')}"
        return state, response

    def _next_question(self, context: DialogContext) -> tuple[State, str]:
        if self.mode == Mode.EAGER and any(context.preferences[slot] is not None for slot in SLOT_ORDER): 
            return self._suggest(context)
        for slot in SLOT_ORDER: 
            if context.preferences[slot] is None:
                return ASK_STATE[slot], self.respond(ASK_TEMPLATE[slot])
        if context.requirements is None:
            return State.ASK_ADDITIONAL, self.respond("askadditional")
        return self._suggest(context)

    def _suggest(self, context: DialogContext) -> tuple[State, str]:
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
            return State.SUGGEST, self.respond("notunderstood")
        names = self._requested_info(utterance)
        if not names:
            return State.GIVE_INFO, self.respond("notunderstood")
        return State.GIVE_INFO, " ".join(self.respond(name, **context.restaurant) for name in names)

    def step(self, state: State, utterance: str, context: DialogContext) -> tuple[State, str, str]:
        utterance = " ".join(re.sub(r"[^a-z0-9 ]", " ", utterance.lower().replace("'", "")).split())
        act = self._closing_act(self.classifier.predict_one(utterance), utterance)
        next_state, system_utterance = self.transition(state, act, utterance, context)
        context.last_utterance = system_utterance
        return next_state, system_utterance, act

    def run(self, log_dir: Path, show_acts: bool = False) -> None:
        state, context = State.WELCOME, DialogContext()
        context.last_utterance = self.respond("welcome")
        logging.info(f"system: {context.last_utterance}")
        turns = [{"system": context.last_utterance}]
        while state != State.TERMINATE:
            try:
                utterance = input("user: ")
            except (EOFError, KeyboardInterrupt):
                break
            state, system_utterance, act = self.step(state, utterance, context)
            if show_acts:
                logging.info(f"        [act={act}, state={state.value}. {state.name}, preferences={context.preferences}]")
            logging.info(f"system: {system_utterance}")

            turns.append({"user": utterance, "act": act, "state": state.name, "preferences": dict(context.preferences), "system": system_utterance})
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"dialog_{datetime.now():%Y%m%d_%H%M%S}.json").open("w", encoding="utf-8") as file:
            json.dump(turns, file, indent=2)
