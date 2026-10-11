import random
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd


PROPERTIES = ("touristic", "assigned_seats", "children", "romantic")


@dataclass(frozen=True)
class Rule:
    id: int
    antecedent: Callable[[pd.Series], bool]
    consequent: str
    value: bool
    reason: str  # The template that explains the rule to the user.


@dataclass
class Candidate:
    restaurant: pd.Series
    reasons: list[str]  # Reason templates for the requested properties, used by the transparent recommendation.


RULES = (
    Rule(1, lambda r: r["pricerange"] == "cheap" and r["food quality"] == "good", "touristic", True,
         "reasoncheapgoodfoodtouristic"),
    Rule(2, lambda r: r["food"] == "romanian", "touristic", False, "reasonromaniannottouristic"),
    Rule(3, lambda r: r["crowdedness"] == "busy", "assigned_seats", True, "reasonbusyassignedseats"),
    Rule(4, lambda r: r["length of stay"] == "long", "children", False, "reasonlongstaynochildren"),
    Rule(5, lambda r: r["crowdedness"] == "busy", "romantic", False, "reasonbusynotromantic"),
    Rule(6, lambda r: r["length of stay"] == "long", "romantic", True, "reasonlongstayromantic"),
)

# Keywords per property; a negation word earlier in the same clause flips the requested value.
REQUIREMENT_PATTERNS = {"touristic": r"touristic|touristy|tourists?",
                        "assigned_seats": r"assigned seat(?:s|ing)?|assigned",
                        "children": r"children|child|kids?|famil(?:y|ies)",
                        "romantic": r"romantic"}
NEGATIONS = {"not", "no", "without", "non", "isnt", "dont", "never"}
CLAUSE_BREAK = r"[,.;!?]|\band\b|\bbut\b"


def reasoning_function(restaurant: pd.Series) -> tuple[dict[str, bool | None], dict[str, list[str]]]:
    conclusions: dict[str, set[bool]] = {name: set() for name in PROPERTIES}
    reasons: dict[str, list[str]] = {name: [] for name in PROPERTIES}
    for rule in RULES:
        if rule.antecedent(restaurant):
            conclusions[rule.consequent].add(rule.value)
            reasons[rule.consequent].append(rule.reason)

    inferred = {name: next(iter(values)) if len(values) == 1 else None for name, values in conclusions.items()}
    return inferred, reasons


def is_contradiction(inferred: dict[str, bool | None], reasons: dict[str, list[str]], name: str) -> bool:
    return inferred[name] is None and len(reasons[name]) > 1


def extract_requirements(utterance: str) -> dict[str, bool]:
    utterance = utterance.replace("'", "")
    found = {}
    for name, pattern in REQUIREMENT_PATTERNS.items():
        match = re.search(rf"\b(?:{pattern})\b", utterance)
        if match is None:
            continue
        clause = re.split(CLAUSE_BREAK, utterance[:match.start()])[-1].split()
        found[name] = not any(word in NEGATIONS for word in clause[-3:])
    return found


def apply_requirements(restaurants: list[pd.Series], requirements: dict[str, bool]
                       ) -> tuple[list[Candidate], list[tuple[pd.Series, str, list[str]]]]:
    candidates, conflicts = [], []
    for restaurant in restaurants:
        inferred, reasons = reasoning_function(restaurant)
        conflict = next((name for name in requirements if is_contradiction(inferred, reasons, name)), None)
        if all(inferred[name] == value for name, value in requirements.items()):
            candidates.append(Candidate(restaurant, [reason for name in requirements for reason in reasons[name]]))
        elif conflict is not None and all(inferred[name] == value for name, value in requirements.items()
                                          if name != conflict):
            conflicts.append((restaurant, conflict, reasons[conflict]))
    return candidates, conflicts


def choose(candidates: list[Candidate]) -> tuple[Candidate | None, list[Candidate]]:
    if not candidates:
        return None, []
    rest = list(candidates)
    return rest.pop(random.randrange(len(rest))), rest
