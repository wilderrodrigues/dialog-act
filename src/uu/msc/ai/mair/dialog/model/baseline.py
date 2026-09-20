"""Rule-based keyword baseline for dialog act classification.

The rules were written by hand after reading the training utterances. They
are checked in order and the first matching rule decides the dialog act;
utterances that match no rule are labelled with the majority class 'inform'.
"""

import re

DEFAULT_ACT = "inform"

# Each rule is (dialog_act, position, keywords).
# position 'start'  -> utterance starts with the keyword
# position 'any'    -> keyword occurs anywhere as whole word(s)
# position 'whole'  -> utterance is exactly the keyword
# Order matters: specific phrases and rare acts come before generic words.
RULES = [
    # "no thank you good bye" is labelled negate, so 'no' precedes 'thank'.
    ("negate", "start", ["no", "nope"]),
    ("thankyou", "any", ["thank"]),
    ("bye", "any", ["bye", "goodbye"]),
    ("restart", "any", ["start over", "start again", "reset"]),
    ("repeat", "any", ["repeat", "again", "back"]),
    ("reqalts", "any", ["anything else", "how about", "what about", "else",
                        "other", "another", "different", "alternative", "next"]),
    ("reqmore", "whole", ["more"]),
    # Preference statements such as "any price range" precede the request rule.
    ("inform", "start", ["any", "cheap", "moderate", "moderately", "expensive",
                         "im looking", "i want", "i need", "id like",
                         "i would like", "looking for"]),
    ("request", "any", ["phone", "telephone", "address", "post code", "postcode",
                        "postal code", "price range", "type of food",
                        "kind of food", "what", "whats", "can i get",
                        "could i get", "may i have", "give me"]),
    ("request", "whole", ["area", "price", "location"]),
    ("confirm", "any", ["is it", "is that", "is this", "does it", "do they"]),
    ("affirm", "any", ["yes", "yeah", "yea", "ye", "yep", "right", "correct",
                       "sure"]),
    ("hello", "any", ["hello", "hi", "halo", "hey"]),
    ("null", "any", ["noise", "sil", "unintelligible", "cough", "breathing",
                     "system", "laughing", "inaudible", "static", "tv"]),
    ("null", "whole", ["okay", "uh", "um", "ah", "oh"]),
    ("deny", "any", ["wrong", "dont want", "do not want", "not that"]),
    ("ack", "start", ["okay", "ok", "kay", "alright", "fine", "good"]),
]


def _compile_rules(rules: list[tuple[str, str, list[str]]]) -> list[tuple[str, re.Pattern]]:
    """Turn every keyword list into one compiled regular expression."""
    compiled = []
    for act, position, keywords in rules:
        alternatives = "|".join(re.escape(keyword) for keyword in keywords)
        if position == "start":
            pattern = rf"^(?:{alternatives})\b"
        elif position == "whole":
            pattern = rf"^(?:{alternatives})$"
        else:
            pattern = rf"\b(?:{alternatives})\b"
        compiled.append((act, re.compile(pattern)))
    return compiled


COMPILED_RULES = _compile_rules(RULES)


def predict_one(utterance: str) -> str:
    """Return the dialog act for a single utterance."""
    utterance = utterance.lower().strip()
    for act, pattern in COMPILED_RULES:
        if pattern.search(utterance):
            return act
    return DEFAULT_ACT


def predict(utterances: list[str]) -> list[str]:
    """Return the predicted dialog act for every utterance in the list."""
    return [predict_one(utterance) for utterance in utterances]
