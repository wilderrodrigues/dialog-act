"""The experiment design: who runs which participant, the goals per conversation and the questionnaire.

Every participant talks to both systems, three conversations each, and fills in the questionnaire after each
system. Persons 1-14 (group A) start with the eager mode, persons 15-28 (group B) with the patient mode. Person
``n + 14`` gets the same goals as person ``n``, in the same column order; only the modes are swapped.
"""
from typing import Any

RESEARCHERS = {
    "Remy": range(1, 8),
    "Cai": range(8, 15),
    "Dexter": range(15, 22),
    "Wilder": range(22, 29),
}
PERSONS = range(1, 29)
GROUP_SIZE = 14

EAGER, PATIENT = "eager", "patient"
LABELS = ("System A", "System B")  # shown instead of the mode, so participants do not know which is which
GOAL_SLOTS = ("area", "pricerange", "food")  # Q1, Q2 and Q3 of every row
CONVERSATIONS_PER_SYSTEM = len(GOAL_SLOTS)

# Group A: the first three columns are the eager goals (EM Q1-Q3), the last three the patient goals (PM Q1-Q3).
# Group B uses the same rows, with PM Q1-Q3 first and EM Q1-Q3 second.
GOALS = {
    1: (("north", "cheap", "thai"), ("east", "moderate", "international")),
    2: (("east", "moderate", "chinese"), ("north", "cheap", "thai")),
    3: (("south", "expensive", "british"), ("east", "moderate", "chinese")),
    4: (("west", "cheap", "italian"), ("south", "expensive", "british")),
    5: (("north", "moderate", "french"), ("west", "cheap", "italian")),
    6: (("east", "expensive", "european"), ("north", "moderate", "french")),
    7: (("south", "cheap", "international"), ("east", "expensive", "european")),
    8: (("west", "moderate", "thai"), ("south", "cheap", "international")),
    9: (("north", "expensive", "chinese"), ("west", "moderate", "thai")),
    10: (("east", "cheap", "british"), ("north", "expensive", "chinese")),
    11: (("south", "moderate", "italian"), ("east", "cheap", "british")),
    12: (("west", "expensive", "french"), ("south", "moderate", "italian")),
    13: (("north", "cheap", "european"), ("west", "expensive", "french")),
    14: (("east", "moderate", "international"), ("north", "cheap", "european")),
}

LIKERT = ("Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree")
AI_USAGE = ("Never", "Rarely", "Sometimes", "Often", "Daily")

# Part A: adapted from the GAAIS; A4 and A5 are reverse scored. Part B: Godspeed likeability (Bartneck et al.,
# 2009). Answers are stored as given (1-5); reverse scoring is left to the analysis.
QUESTIONNAIRE = [
    {"id": 1, "code": "A1", "part": "A", "reversed": False,
     "text": "I feel sympathetic towards the way the recommender system assisted me in my search for a restaurant."},
    {"id": 2, "code": "A2", "part": "A", "reversed": False,
     "text": "Interacting with this recommender system was enjoyable."},
    {"id": 3, "code": "A3", "part": "A", "reversed": False,
     "text": "This system made my search for a restaurant more pleasant."},
    {"id": 4, "code": "A4", "part": "A", "reversed": True, "text": "I found this recommender system unsettling."},
    {"id": 5, "code": "A5", "part": "A", "reversed": True, "text": "Interacting with this system made me uncomfortable."},
    {"id": 6, "code": "B1", "part": "B", "reversed": False, "left": "Dislike", "right": "Like"},
    {"id": 7, "code": "B2", "part": "B", "reversed": False, "left": "Unfriendly", "right": "Friendly"},
    {"id": 8, "code": "B3", "part": "B", "reversed": False, "left": "Unkind", "right": "Kind"},
    {"id": 9, "code": "B4", "part": "B", "reversed": False, "left": "Unpleasant", "right": "Pleasant"},
    {"id": 10, "code": "B5", "part": "B", "reversed": False, "left": "Awful", "right": "Nice"},
]
QUESTION_IDS = frozenset(question["id"] for question in QUESTIONNAIRE)


def researcher_of(person: int) -> str:
    return next(name for name, persons in RESEARCHERS.items() if person in persons)


def group_of(person: int) -> str:
    return "A" if person <= GROUP_SIZE else "B"


def describe(slot: str, value: str) -> str:
    """The goal as the participant sees it: "try to find <a restaurant ...> that you like"."""
    if slot == "area":
        restaurant = f"a restaurant in the {value} part of town"
    elif slot == "pricerange":
        restaurant = {"cheap": "a cheap restaurant", "moderate": "a moderately priced restaurant",
                      "expensive": "an expensive restaurant"}[value]
    else:
        article = "an" if value[0] in "aeiou" and not value.startswith("eu") else "a"
        restaurant = f"{article} {value.title()} restaurant"
    return f"try to find {restaurant} that you like"


def plan(person: int) -> list[dict[str, Any]]:
    """The two systems a person talks to, in order, each with its three goals."""
    first, second = GOALS[(person - 1) % GROUP_SIZE + 1]
    modes = (EAGER, PATIENT) if group_of(person) == "A" else (PATIENT, EAGER)
    return [{"mode": mode, "label": label, "order": order,
             "goals": [{"slot": slot, "value": value, "text": describe(slot, value)}
                       for slot, value in zip(GOAL_SLOTS, values)]}
            for order, (mode, label, values) in enumerate(zip(modes, LABELS, (first, second)), start=1)]


TEST_GOAL = "test"


def test_plan(person: int) -> list[dict[str, Any]]:
    """A researcher's trial run: one conversation per system (in the mode order of ``person``), goal "test"."""
    return [{**system, "goals": [{"slot": TEST_GOAL, "value": TEST_GOAL, "text": TEST_GOAL}]}
            for system in plan(person)]


def overview() -> dict[str, Any]:
    """Everything the researcher screen needs: who runs whom, and every person's plan."""
    return {
        "researchers": {name: list(persons) for name, persons in RESEARCHERS.items()},
        "persons": {person: {"group": group_of(person), "plan": plan(person)} for person in PERSONS},
        "questionnaire": QUESTIONNAIRE,
        "likert": LIKERT,
        "ai_usage": AI_USAGE,
    }
