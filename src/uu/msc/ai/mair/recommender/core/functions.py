import json
import random
import re
from pathlib import Path
import Levenshtein
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from uu.msc.ai.mair.dialog.core.encoders import (EMBED_LEN, Encoder, SimpleEncoder, FrozenDistilBertEncoder,
                                                 DistilBertTokenEncoder)

PREFERENCES = ("pricerange", "area", "food")
STOPWORDS = {"i", "want", "wanna", "looking", "restaurant", "place", "please", "food", "part", "town",
             "that", "serves", "serving", "with", "have", "find", "would", "like", "need", "something",
             "somewhere", "what", "about", "there", "also", "thanks", "hello","a"}

DONT_CARE = "dontcare"
UNKNOWN = "unknown"

VALUE_VECTORS = {}
DONT_CARE_PATTERNS = ("any", "dont care", "doesnt matter", "does not matter", "whatever")
DONT_CARE_HINTS = {"pricerange": ("price",), "area": ("area", "part", "town"), "food": ("food", "type", "kind")}
SYNONYMS = {"center": "centre", "moderately": "moderate", "cheaper": "cheap", "pricey": "expensive","fancy": "expensive", "upscale": "expensive", "luxurious": "expensive",
            "affordable": "cheap", "inexpensive": "cheap", "budget": "cheap"}


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


def load_slot_csv(csv_path: Path):
    data = pd.read_csv(csv_path)
    slotdict = {}
    for item in PREFERENCES:
        columnitems = data[f"{item}"].dropna().tolist()
        slotdict[item] = set(columnitems) 
    return slotdict

def parse_sentence(sentence: str) -> list[str]:
    words = re.findall(r"[a-z']+", sentence.lower())
    return [word for word in words if word not in STOPWORDS]


def levenshtein_distance(sentence: str, document: dict, matched: dict, max_distance: int = 2) -> dict:

    parsesentence = parse_sentence(sentence)
 
    for i in range(len(parsesentence)):
        for value in matched.values():
            if parsesentence[i] in value.split(" "):
                parsesentence[i] = "#"
 
    matches = {}
    for slots in PREFERENCES:
        if slots in matched:
            continue
 
        best_value = None
        best_distance = max_distance + 1
        for value in document[slots]:
            n = len(value.split(" "))
            for i in range(len(parsesentence) - n + 1):
                window = parsesentence[i:i + n]
                if "#" in window:
                    continue
                if n == 1 and len(window[0]) < 4: 
                    continue
 
                distance = Levenshtein.distance(" ".join(window), value)
                if distance < best_distance:
                    best_distance = distance
                    best_value = value
 
        if best_value is not None:
            matches[slots] = (best_value, best_distance)
    return matches
 




#TODO it is really bad the encoding always guesses wrong something wrong with bert?
def semantic_similarity(sentence: str, document: dict, matched: dict, encoder, threshold: float = 0.8) -> dict:

    parsesentence = parse_sentence(sentence)

    for i in range(len(parsesentence)):
        for value in matched.values():
            if parsesentence[i] in value.split(" "):
                parsesentence[i] = "#"

    words = [word for word in parsesentence if word != "#" and len(word) >= 3]
    if not words:
        return {}

    token_vectors = encoder.encode_sentence(words)
    lengths = (np.abs(token_vectors).sum(-1) > 0).sum(1)
    word_vectors = np.stack([vectors[1: n - 1].mean(0) for vectors, n in zip(token_vectors, lengths)])
 
    matches = {}
    for slots in PREFERENCES:
        if slots in matched:
            continue
 
        if slots not in VALUE_VECTORS:
            values = sorted(document[slots])
            token_vectors = encoder.encode_sentence(values)
            lengths = (np.abs(token_vectors).sum(-1) > 0).sum(1)
            VALUE_VECTORS[slots] = (values, np.stack([vectors[1: n - 1].mean(0) for vectors, n in zip(token_vectors, lengths)]))
        
        values, value_vectors = VALUE_VECTORS[slots]
 
        similarities = cosine_similarity(word_vectors, value_vectors)   
        best_scores = similarities.max(axis=0)                          
        best_value = best_scores.argmax()                               
 
        if best_scores[best_value] >= threshold:
            matches[slots] = (values[best_value], float(best_scores[best_value]))
    return matches


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

def smalltest():
    load_slot_csv(Path("data/restaurant_info.csv"))
    sentence = "I want a fancy restaurant serving europesan food"
    data = load_slot_csv(Path("data/restaurant_info.csv"))
    matches = keyword_matching(sentence,data)
    leven = levenshtein_distance(sentence,data,matches)
    encoder = FrozenDistilBertEncoder()
    encoder.init_tokenizer(None)
    seman = semantic_similarity(sentence,data,matches,encoder)
    return 0
