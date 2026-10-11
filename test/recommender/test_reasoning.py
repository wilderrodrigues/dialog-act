import pandas as pd
import pytest

from uu.msc.ai.mair.recommender.core.reasoning import (apply_requirements, extract_requirements, is_contradiction,
                                                       reasoning_function)


def restaurant(pricerange="moderate", food="british", quality="mediocre", crowdedness="not busy",
               stay="short") -> pd.Series:
    return pd.Series({"restaurantname": "test", "pricerange": pricerange, "food": food, "food quality": quality,
                      "crowdedness": crowdedness, "length of stay": stay})


def test_no_rule_fires_leaves_everything_unknown() -> None:
    inferred, reasons = reasoning_function(restaurant())
    assert set(inferred.values()) == {None}
    assert not any(reasons.values())


def test_cheap_and_good_food_is_touristic() -> None:
    inferred, reasons = reasoning_function(restaurant(pricerange="cheap", quality="good"))
    assert inferred["touristic"] is True
    assert reasons["touristic"] == ["reasoncheapgoodfoodtouristic"]


def test_busy_restaurant() -> None:
    inferred, _ = reasoning_function(restaurant(crowdedness="busy"))
    assert inferred["assigned_seats"] is True and inferred["romantic"] is False


def test_long_stay_restaurant() -> None:
    inferred, _ = reasoning_function(restaurant(stay="long"))
    assert inferred["children"] is False and inferred["romantic"] is True


@pytest.mark.parametrize("kwargs, name", [({"crowdedness": "busy", "stay": "long"}, "romantic"),
                                          ({"pricerange": "cheap", "quality": "good", "food": "romanian"},
                                           "touristic")])
def test_contradicting_rules_give_unknown(kwargs, name) -> None:
    inferred, reasons = reasoning_function(restaurant(**kwargs))
    assert inferred[name] is None
    assert is_contradiction(inferred, reasons, name)


@pytest.mark.parametrize("utterance, expected", [
    ("i want a romantic place", {"romantic": True}),
    ("not romantic and good for kids", {"romantic": False, "children": True}),
    ("no children, but touristic", {"children": False, "touristic": True}),
    ("i don't want assigned seats", {"assigned_seats": False}),
    ("no", {}),
])
def test_extract_requirements(utterance, expected) -> None:
    assert extract_requirements(utterance) == expected


def test_apply_requirements_filters_and_reports_conflicts() -> None:
    romantic, busy, conflicting = restaurant(stay="long"), restaurant(crowdedness="busy"),         restaurant(crowdedness="busy", stay="long")
    candidates, conflicts = apply_requirements([romantic, busy, conflicting], {"romantic": True})
    assert [c.restaurant is romantic for c in candidates] == [True]
    assert candidates[0].reasons == ["reasonlongstayromantic"]
    assert [(r is conflicting, name) for r, name, _ in conflicts] == [(True, "romantic")]
