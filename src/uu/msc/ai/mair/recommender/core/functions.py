import json
import random
import re
from pathlib import Path

import pandas as pd
import numpy.typing as npt

PREFERENCES = ("pricerange", "area", "food")
DONT_CARE = "dontcare"
UNKNOWN = "unknown"

DONT_CARE_PATTERNS = ("any", "dont care", "doesnt matter", "does not matter", "whatever")
DONT_CARE_HINTS = {"pricerange": ("price",), "area": ("area", "part", "town"), "food": ("food", "type", "kind")}
SYNONYMS = {"center": "centre", "moderately": "moderate", "cheaper": "cheap", "pricey": "expensive"}


def keyword_matching(utterance: str, document: pd.DataFrame, asked_slot: str | None = None) -> dict[str, str]:
    """Finds the preferences named in the utterance; 'any'/'dont care' goes to the slot it names, else the asked one."""
    words = " ".join(SYNONYMS.get(word, word) for word in utterance.split())
    found = {}
    for slot in PREFERENCES:
        # Longest values first, so 'modern european' wins over 'european'.
        values = sorted((value for value in document[slot].unique() if value != UNKNOWN), key=len, reverse=True)
        match = next((value for value in values if re.search(rf"\b{re.escape(value)}\b", words)), None)
        if match is not None:
            found[slot] = match

    if any(re.search(rf"\b{pattern}\b", words) for pattern in DONT_CARE_PATTERNS):
        hinted = [slot for slot, hints in DONT_CARE_HINTS.items() if any(hint in words for hint in hints)]
        for slot in hinted or ([asked_slot] if asked_slot else []):
            found.setdefault(slot, DONT_CARE)
    return found


def levenshtein_distance(word: str, document: pd.DataFrame) -> list[str]:
    pass


def cosine_similarity(word: str, document: npt.NDArray) -> tuple[list[str], list[float]]:
    pass


def load_restaurants(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path, keep_default_na=False).replace("", UNKNOWN)


def find_restaurants(restaurants: pd.DataFrame,
                     preferences: dict[str, str | None]) -> tuple[pd.Series | None, list[pd.Series]]:
    matches = restaurants
    for slot in PREFERENCES:
        value = preferences.get(slot)
        if value is not None and value != DONT_CARE: # restaurants are only recommended when the area does not matter.
            matches = matches[matches[slot] == value]

    if matches.empty:
        return None, []
    rows = [row for _, row in matches.iterrows()]
    chosen = rows.pop(random.randrange(len(rows)))
    return chosen, rows


def load_templates(data_path: Path) -> dict[str, str]:
    with data_path.open(encoding="utf-8") as file:
        return json.load(file)


def generate_response(templates: dict[str, str], name: str, transparent: bool = False,
                      reasons: list[str] | None = None, **slots: str) -> str:
    if transparent and reasons and f"{name}withreasons" in templates:
        explanation = " and ".join(templates[reason].format(**slots) for reason in reasons)
        return templates[f"{name}withreasons"].format(reasons=explanation, **slots)
    return templates[name].format(**slots)
