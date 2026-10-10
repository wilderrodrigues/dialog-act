import json
import random
from pathlib import Path
import Levenshtein
import re
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

    

def keyword_matching(sentence: str, document: dict) -> list[str]:

    parsesentence = parse_sentence(sentence)
    max_words = max(len(value.split(" ")) for slots in PREFERENCES for value in document[slots])
    matches = {}
 
    for n in range(max_words, 0, -1):
        for slots in PREFERENCES:
            if slots in matches:
                continue
            for value in document[slots]:
                valuewords = value.split(" ")
                if len(valuewords) != n:
                    continue
                for i in range(len(parsesentence) - n + 1):
                    if parsesentence[i:i + n] == valuewords:
                        matches[slots] = value
                        parsesentence[i:i + n] = ["#"] * n
                        break
    return matches




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
def semantic_similarity(sentence: str, document: dict, matched: dict, encoder, threshold: float = 0.75) -> dict:

    parsesentence = parse_sentence(sentence)

    for i in range(len(parsesentence)):
        for value in matched.values():
            if parsesentence[i] in value.split(" "):
                parsesentence[i] = "#"

    words = [word for word in parsesentence if word != "#" and len(word) >= 3]
    if not words:
        return {}

    word_vectors = encoder.encode_sentences(words)
 
    matches = {}
    for slots in PREFERENCES:
        if slots in matched:
            continue
 
        if slots not in VALUE_VECTORS:
            values = sorted(document[slots])
            VALUE_VECTORS[slots] = (values, encoder.encode_sentences(values))
        
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
