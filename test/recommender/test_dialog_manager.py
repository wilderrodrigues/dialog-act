from uu import get_root
from uu.msc.ai.mair.dialog.model.rule_based_classifier import RuleBasedModel
from uu.msc.ai.mair.recommender.core.dialog_manager import DialogContext, DialogManager, State
from uu.msc.ai.mair.recommender.core.functions import DONT_CARE


def make_manager() -> DialogManager:
    return DialogManager(classifier=RuleBasedModel(), restaurants_path=get_root() / "data" / "restaurant_info.csv",
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


def test_all_preferences_in_one_utterance_recommends_a_restaurant() -> None:
    context = DialogContext()
    state, utterance, _ = make_manager().step(State.WELCOME, "cheap italian food in the centre", context)
    assert state == State.SUGGEST
    assert context.restaurant["food"] == "italian"
    assert utterance.startswith(f"I recommend {context.restaurant['restaurantname']}")


def test_dontcare_fills_the_asked_slot() -> None:
    context = DialogContext()
    make_manager().step(State.ASK_PRICE, "i dont care", context)
    assert context.preferences["pricerange"] == DONT_CARE


def test_alternatives_run_out() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    for _ in range(len(context.alternatives)):
        _, utterance, _ = manager.step(State.SUGGEST, "is there anything else", context)
        assert utterance.startswith("How about")
    _, utterance, _ = manager.step(State.SUGGEST, "is there anything else", context)
    assert utterance == manager.templates["noalternatives"]


def test_request_gives_info_and_bye_terminates() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    state, utterance, _ = manager.step(State.SUGGEST, "what is the phone number", context)
    assert state == State.GIVE_INFO
    assert context.restaurant["phone"] in utterance
    state, _, _ = manager.step(state, "thank you good bye", context)
    assert state == State.TERMINATE
