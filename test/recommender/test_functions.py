from string import Formatter

import pytest

from uu import get_root
from uu.msc.ai.mair.recommender.core.functions import (DONT_CARE, find_restaurants, generate_response, keyword_matching,
                                                      load_restaurants, load_templates)

RESTAURANTS_FILE = get_root() / "test" / "resources" / "test_restaurants.csv"
TEMPLATES_FILE = get_root() / "data" / "templates.json"


@pytest.fixture(scope="module")
def restaurants():
    return load_restaurants(RESTAURANTS_FILE)


@pytest.fixture(scope="module")
def templates() -> dict[str, str]:
    return load_templates(TEMPLATES_FILE)


def names(chosen, alternatives) -> list[str]:
    return [chosen["restaurantname"]] + [row["restaurantname"] for row in alternatives]


def test_empty_fields_are_spelled_out(restaurants) -> None:
    unknown_thai = restaurants[restaurants["restaurantname"] == "the unknown thai"].iloc[0]
    assert unknown_thai["area"] == "unknown"
    assert unknown_thai["phone"] == "unknown"


def test_every_given_preference_filters(restaurants) -> None:
    chosen, alternatives = find_restaurants(restaurants, {"pricerange": "cheap", "area": "north", "food": "thai"})
    assert sorted(names(chosen, alternatives)) == ["the first thai", "the second thai"]


def test_dontcare_and_missing_preferences_do_not_filter(restaurants) -> None:
    chosen, alternatives = find_restaurants(restaurants, {"area": "dontcare", "food": "thai"})
    assert len(names(chosen, alternatives)) == 4


def test_unknown_area_only_matches_when_the_area_does_not_matter(restaurants) -> None:
    chosen, alternatives = find_restaurants(restaurants, {"area": "north", "food": "thai"})
    assert "the unknown thai" not in names(chosen, alternatives)

    chosen, alternatives = find_restaurants(restaurants, {"area": "dontcare", "food": "thai"})
    assert "the unknown thai" in names(chosen, alternatives)


def test_alternatives_keep_file_order_without_the_chosen_one(restaurants) -> None:
    in_file_order = ["the first thai", "the second thai", "the expensive thai", "the unknown thai"]
    for _ in range(20):
        chosen, alternatives = find_restaurants(restaurants, {"food": "thai"})
        assert [row["restaurantname"] for row in alternatives] == [name for name in in_file_order
                                                                   if name != chosen["restaurantname"]]


def test_no_match_returns_nothing(restaurants) -> None:
    assert find_restaurants(restaurants, {"food": "korean"}) == (None, [])


def test_every_template_can_be_filled(templates: dict[str, str]) -> None:
    for name, template in templates.items():
        fields = {field for _, field, _, _ in Formatter().parse(template) if field}
        assert template.format(**{field: "x" for field in fields})


def test_template_variables_are_filled(templates: dict[str, str]) -> None:
    response = generate_response(templates, "confirmfoodtype", givenfoodtype="itlian", correctedfoodtype="italian")
    assert response == "I did not recognize itlian. Did you mean italian?"


def test_unknown_template_or_missing_variable_raises(templates: dict[str, str]) -> None:
    with pytest.raises(KeyError):
        generate_response(templates, "nosuchtemplate")
    with pytest.raises(KeyError):
        generate_response(templates, "confirmfoodtype", givenfoodtype="itlian")


def test_recommendation_is_filled_from_a_restaurant(restaurants, templates: dict[str, str]) -> None:
    chosen, _ = find_restaurants(restaurants, {"food": "italian"})
    assert generate_response(templates, "recommend", **chosen) == "I recommend the cheap italian, a cheap italian restaurant in the south part of town."


def test_transparency_explains_the_recommendation(restaurants, templates: dict[str, str]) -> None:
    chosen, _ = find_restaurants(restaurants, {"food": "italian"})
    response = generate_response(templates, "recommend", transparent=True,
                                 reasons=["reasonbusyassignedseats", "reasonlongstayromantic"], **chosen)
    assert response.endswith("because it is busy, so it has assigned seats "
                             "and it allows a long stay, which makes it romantic.")


def test_transparency_without_reasons_or_variant_uses_the_plain_template(restaurants,
                                                                         templates: dict[str, str]) -> None:
    chosen, _ = find_restaurants(restaurants, {"food": "italian"})
    assert generate_response(templates, "recommend", transparent=True, reasons=[], **chosen) ==            generate_response(templates, "recommend", **chosen)
    assert generate_response(templates, "askarea", transparent=True, reasons=["reasonbusyassignedseats"]) ==            templates["askarea"]


def test_keyword_matching_finds_every_preference(restaurants) -> None:
    assert keyword_matching("cheap thai food in the north", restaurants) == {"pricerange": "cheap", "area": "north",
                                                                             "food": "thai"}


def test_keyword_matching_uses_whole_words_and_synonyms(restaurants) -> None:
    assert keyword_matching("northern food", restaurants) == {}
    assert keyword_matching("something cheaper", restaurants) == {"pricerange": "cheap"}


def test_keyword_matching_never_matches_unknown(restaurants) -> None:
    assert keyword_matching("unknown", restaurants) == {}


def test_dontcare_goes_to_the_named_or_asked_slot(restaurants) -> None:
    assert keyword_matching("any part of town", restaurants) == {"area": DONT_CARE}
    assert keyword_matching("i dont care", restaurants, asked_slot="pricerange") == {"pricerange": DONT_CARE}
    assert keyword_matching("i dont care", restaurants) == {}
