import json
import random
from pathlib import Path

import pandas as pd
import numpy.typing as npt

PREFERENCES = ("pricerange", "area", "food")
DONT_CARE = "dontcare"
UNKNOWN = "unknown"

def keyword_matching(word: str, document: pd.DataFrame) -> list[str]:
    pass


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
