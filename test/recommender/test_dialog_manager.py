from uu import get_root
from uu.msc.ai.mair.dialog.model.rule_based_classifier import RuleBasedModel
from uu.msc.ai.mair.recommender.core.dialog_manager import DialogContext, DialogManager, State
from uu.msc.ai.mair.recommender.core.functions import DONT_CARE


def make_manager() -> DialogManager:
    return DialogManager(classifier=RuleBasedModel(), restaurants_path=get_root() / "data" / "restaurant_info_extended.csv",
                         templates_path=get_root() / "data" / "templates.json")


def test_welcome_without_preferences_asks_area() -> None:
    state, _, _ = make_manager().step(State.WELCOME, "hello", DialogContext())
    assert state == State.ASK_AREA


def test_missing_slots_are_asked_in_order() -> None:
    manager, context = make_manager(), DialogContext()
    state, _, _ = manager.step(State.WELCOME, "i want italian food", context)
    assert state == State.ASK_AREA
    state, _, _ = manager.step(state, "the south part of town", context)
    assert state == State.ASK_PRICE
    assert context.preferences == {"area": "south", "food": "italian", "pricerange": None}


def test_ask_state_loops_without_new_information() -> None:
    manager = make_manager()
    state, utterance, _ = manager.step(State.ASK_FOOD, "yes", DialogContext())
    assert state == State.ASK_FOOD
    assert utterance == manager.templates["askfoodtype"]


def test_all_preferences_in_one_utterance_asks_additional_requirements() -> None:
    state, _, _ = make_manager().step(State.WELCOME, "cheap italian food in the centre", DialogContext())
    assert state == State.ASK_ADDITIONAL


def test_no_additional_requirements_recommends_a_restaurant() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    state, utterance, _ = manager.step(State.ASK_ADDITIONAL, "no", context)
    assert state == State.SUGGEST
    assert context.requirements == {}
    assert context.restaurant["food"] == "italian"
    assert utterance.startswith(f"I recommend {context.restaurant['restaurantname']}")


def test_dontcare_fills_the_asked_slot() -> None:
    context = DialogContext()
    make_manager().step(State.ASK_PRICE, "i dont care", context)
    assert context.preferences["pricerange"] == DONT_CARE


def test_alternatives_run_out() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    manager.step(State.ASK_ADDITIONAL, "no", context)
    for _ in range(len(context.alternatives)):
        _, utterance, _ = manager.step(State.SUGGEST, "is there anything else", context)
        assert utterance.startswith("How about")
    _, utterance, _ = manager.step(State.SUGGEST, "is there anything else", context)
    assert utterance == manager.templates["noalternatives"]


def test_request_gives_info_and_bye_terminates() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    manager.step(State.ASK_ADDITIONAL, "no", context)
    state, utterance, _ = manager.step(State.SUGGEST, "what is the phone number", context)
    assert state == State.GIVE_INFO
    assert context.restaurant["phone"] in utterance
    state, _, _ = manager.step(state, "thank you good bye", context)
    assert state == State.TERMINATE


def test_requirements_in_the_first_utterance_skip_the_question() -> None:
    manager, context = make_manager(), DialogContext()
    state, _, _ = manager.step(State.WELCOME, "a romantic expensive restaurant in the centre, any food", context)
    assert state == State.SUGGEST
    assert context.requirements == {"romantic": True}


def test_requirement_filters_and_is_explained() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "expensive food in the centre, any kind", context)
    state, utterance, _ = manager.step(State.ASK_ADDITIONAL, "it should be romantic", context)
    assert state == State.SUGGEST
    assert context.restaurant["length of stay"] == "long" and context.restaurant["crowdedness"] == "not busy"
    assert manager.templates["reasonlongstayromantic"] in utterance


def test_reasoning_can_be_hidden() -> None:
    manager, context = make_manager(), DialogContext()
    manager.transparent = False
    manager.step(State.WELCOME, "expensive food in the centre, any kind", context)
    _, utterance, _ = manager.step(State.ASK_ADDITIONAL, "romantic", context)
    assert "because" not in utterance


def test_contradiction_is_explained_when_nothing_else_matches() -> None:
    manager, context = make_manager(), DialogContext()
    # the gardenia: the only romanian restaurant, busy and long stay -> romantic is both True and False.
    manager.step(State.WELCOME, "moderate romanian food in the west", context)
    state, utterance, _ = manager.step(State.ASK_ADDITIONAL, "romantic please", context)
    assert state == State.SUGGEST and context.restaurant is None
    assert "the gardenia" in utterance and "contradict" in utterance
