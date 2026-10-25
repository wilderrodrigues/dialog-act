import pytest

from uu import get_root
from uu.msc.ai.mair.recommender.core.dialog_manager import DialogContext, DialogManager, State, Mode
from uu.msc.ai.mair.recommender.core.functions import DONT_CARE
from uu.msc.ai.mair.recommender.core.inference_engine import LogRegPredictor
from uu.msc.ai.mair.recommender.core.storage.persistent import ExperimentData


class StubPredictor:
    def __init__(self, act: str = "inform") -> None:
        self.act = act

    def predict_one(self, utterance: str) -> str:
        return self.act


def make_manager(act: str = "inform", mode: Mode = Mode.PATIENT) -> DialogManager:
    return DialogManager(
        classifier=StubPredictor(act),
        restaurants_path=get_root() / "data" / "restaurant_info_extended.csv",
        templates_path=get_root() / "data" / "templates.json",
        experiment_storage=ExperimentData(get_root() / "data" / "experiment.json"),
        mode=mode
    )


def test_welcome_null_starts_slot_collection() -> None:
    state, response, _ = make_manager("null").step(State.WELCOME, "", DialogContext())
    assert state == State.ASK_AREA
    assert response == make_manager().templates["askarea"]


def test_welcome_inform_without_preferences_asks_area() -> None:
    state, _, _ = make_manager().step(State.WELCOME, "hello there", DialogContext())
    assert state == State.ASK_AREA


def test_missing_slots_are_asked_in_order() -> None:
    manager, context = make_manager(), DialogContext()
    state, _, _ = manager.step(State.WELCOME, "i want italian food", context)
    assert state == State.ASK_AREA
    state, _, _ = manager.step(state, "the south part of town", context)
    assert state == State.ASK_PRICE
    assert context.preferences == {"area": "south", "food": "italian", "pricerange": None}


@pytest.mark.parametrize("state,utterance,expected_template", [
    (State.ASK_AREA, "yes", "askarea"),
    (State.ASK_AREA, "", "askarea"),
    (State.ASK_FOOD, "yes", "askfoodtype"),
    (State.ASK_FOOD, "", "askfoodtype"),
    (State.ASK_PRICE, "yes", "askpricerange"),
    (State.ASK_PRICE, "", "askpricerange"),
])
def test_affirm_and_null_repeat_the_current_slot_question(state: State, utterance: str,
                                                          expected_template: str) -> None:
    manager = make_manager("affirm" if utterance else "null")
    next_state, response, _ = manager.step(state, utterance, DialogContext())
    assert next_state == state
    assert response == manager.templates[expected_template]


@pytest.mark.parametrize("state", [State.ASK_AREA, State.ASK_FOOD, State.ASK_PRICE])
@pytest.mark.parametrize("act", ["request", "hello", "ack", "deny", "confirm", "reqmore", "unknown"])
def test_unsupported_slot_acts_recover_without_changing_context(state: State, act: str) -> None:
    manager, context = make_manager(act), DialogContext()
    before = dict(context.preferences)
    next_state, response, _ = manager.step(state, "north italian cheap", context)
    assert next_state == state
    assert response.startswith(manager.templates["notunderstood"])
    assert context.preferences == before


def test_unrecognized_inform_recovers_and_repeats_question() -> None:
    manager = make_manager("inform")
    state, response, _ = manager.step(State.ASK_FOOD, "hello there", DialogContext())
    assert state == State.ASK_FOOD
    assert response == f"{manager.templates['notunderstood']} {manager.templates['askfoodtype']}"


def test_valid_information_advances_even_when_it_fills_a_different_slot() -> None:
    manager, context = make_manager(), DialogContext()
    state, _, _ = manager.step(State.ASK_AREA, "cheap italian food", context)
    assert state == State.ASK_AREA
    assert context.preferences == {"area": None, "food": "italian", "pricerange": "cheap"}


def test_dontcare_fills_the_asked_slot() -> None:
    context = DialogContext()
    make_manager().step(State.ASK_PRICE, "i dont care", context)
    assert context.preferences["pricerange"] == DONT_CARE


def test_all_preferences_in_one_utterance_asks_additional_requirements() -> None:
    state, _, _ = make_manager().step(State.WELCOME, "cheap italian food in the centre", DialogContext())
    assert state == State.ASK_ADDITIONAL


def test_early_requirements_skip_the_additional_question() -> None:
    manager, context = make_manager(), DialogContext()
    state, _, _ = manager.step(State.WELCOME,
                               "a romantic expensive restaurant in the centre, any food", context)
    assert state == State.SUGGEST
    assert context.requirements == {"romantic": True}


def test_no_additional_requirements_proceeds_to_recommendation() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    manager.classifier.act = "negate"
    state, utterance, _ = manager.step(State.ASK_ADDITIONAL, "no", context)
    assert state == State.SUGGEST
    assert context.requirements == {}
    assert context.restaurant["food"] == "italian"
    assert utterance.startswith(f"I recommend {context.restaurant['restaurantname']}")


def test_affirm_at_additional_question_recovers_without_mutating_context() -> None:
    manager, context = make_manager("affirm"), DialogContext(requirements=None)
    before = dict(context.preferences)
    state, response, _ = manager.step(State.ASK_ADDITIONAL, "yes", context)
    assert state == State.ASK_ADDITIONAL
    assert response == f"{manager.templates['notunderstood']} {manager.templates['askadditional']}"
    assert context.preferences == before
    assert context.requirements is None


def test_inform_dontcare_at_additional_question_means_no_requirements() -> None:
    manager, context = make_manager("inform"), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"}
    )
    state, _, _ = manager.step(State.ASK_ADDITIONAL, "any requirements", context)
    assert state == State.SUGGEST
    assert context.requirements == {}


def test_inform_without_requirement_at_additional_question_does_not_mutate_context() -> None:
    manager, context = make_manager("inform"), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"}
    )
    state, _, _ = manager.step(State.ASK_ADDITIONAL, "hello", context)
    assert state == State.ASK_ADDITIONAL
    assert context.requirements is None


@pytest.mark.parametrize("act", ["affirm", "request", "reqalts", "hello", "ack", "unknown"])
def test_unsupported_additional_question_acts_recover_without_context_changes(act: str) -> None:
    manager, context = make_manager(act), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"}
    )
    before = dict(context.preferences)
    state, response, _ = manager.step(State.ASK_ADDITIONAL, "yes", context)
    assert state == State.ASK_ADDITIONAL
    assert response.startswith(manager.templates["notunderstood"])
    assert context.preferences == before
    assert context.requirements is None


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
    manager.step(State.WELCOME, "moderate romanian food in the west", context)
    state, utterance, _ = manager.step(State.ASK_ADDITIONAL, "romantic please", context)
    assert state == State.SUGGEST and context.restaurant is None
    assert "the gardenia" in utterance and "contradict" in utterance


def test_reqalts_selects_alternatives_then_reports_when_exhausted() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    manager.classifier.act = "negate"
    manager.step(State.ASK_ADDITIONAL, "no", context)
    manager.classifier.act = "reqalts"
    while context.alternatives:
        state, utterance, _ = manager.step(State.SUGGEST, "anything else", context)
        assert state == State.SUGGEST
        assert utterance.startswith("How about")
    _, utterance, _ = manager.step(State.SUGGEST, "anything else", context)
    assert utterance == manager.templates["noalternatives"]


def test_reqalts_with_new_preference_runs_a_new_search() -> None:
    manager, context = make_manager(), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"}, requirements={}
    )
    manager.classifier.act = "reqalts"
    state, utterance, _ = manager.step(State.SUGGEST, "how about chinese", context)
    assert state == State.SUGGEST
    assert context.preferences["food"] == "chinese"
    assert context.restaurant["food"] == "chinese"
    assert utterance.startswith("I recommend") or utterance == manager.templates["nomatch"]


def test_inform_can_change_preferences_after_recommendation() -> None:
    manager, context = make_manager(), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"}, requirements={}
    )
    state, _, _ = manager.step(State.SUGGEST, "actually chinese", context)
    assert state == State.SUGGEST
    assert context.preferences["food"] == "chinese"


def test_request_gives_info_and_thankyou_terminates() -> None:
    manager, context = make_manager(), DialogContext()
    manager.step(State.WELCOME, "cheap italian food in the centre", context)
    manager.classifier.act = "negate"
    manager.step(State.ASK_ADDITIONAL, "no", context)
    manager.classifier.act = "request"
    state, utterance, _ = manager.step(State.SUGGEST, "what is the phone number", context)
    assert state == State.GIVE_INFO
    assert context.restaurant["phone"] in utterance
    manager.classifier.act = "request"
    state, utterance, _ = manager.step(state, "address and post code", context)
    assert state == State.GIVE_INFO
    assert context.restaurant["addr"] in utterance
    assert context.restaurant["postcode"] in utterance
    manager.classifier.act = "thankyou"
    state, _, _ = manager.step(state, "thank you", context)
    assert state == State.TERMINATE


def test_request_without_selected_restaurant_recovers() -> None:
    manager, context = make_manager("request"), DialogContext()
    state, response, _ = manager.step(State.SUGGEST, "phone number", context)
    assert state == State.SUGGEST
    assert response == manager.templates["notunderstood"]


@pytest.mark.parametrize("state", [State.SUGGEST, State.GIVE_INFO])
@pytest.mark.parametrize("act", ["affirm", "negate", "hello", "ack", "deny", "confirm", "reqmore", "unknown"])
def test_unsupported_recommendation_acts_recover_without_context_changes(state: State, act: str) -> None:
    manager, context = make_manager(act), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"}, requirements={}
    )
    before = dict(context.preferences)
    next_state, response, _ = manager.step(state, "yes", context)
    assert next_state == state
    assert response == manager.templates["notunderstood"]
    assert context.preferences == before


def test_restart_resets_all_dialog_context() -> None:
    manager, context = make_manager("restart"), DialogContext(
        preferences={"area": "centre", "food": "italian", "pricerange": "cheap"},
        requirements={"romantic": True},
    )
    state, _, _ = manager.step(State.SUGGEST, "start again", context)
    assert state == State.WELCOME
    assert context.preferences == {"area": None, "food": None, "pricerange": None}
    assert context.requirements is None
    assert context.restaurant is None
    assert context.alternatives == []


def test_logistic_regression_predictor_smoke_conversation() -> None:
    predictor = LogRegPredictor(get_root() / "model-zoo" / "Logistic regression" / "log_reg_simple_grouped.pkl")
    manager = DialogManager(
        predictor,
        get_root() / "data" / "restaurant_info_extended.csv",
        get_root() / "data" / "templates.json",
        experiment_storage=ExperimentData(get_root() / "data" / "experiment.json"),
    )
    context = DialogContext()
    state, _, _ = manager.step(State.WELCOME, "i want a cheap italian restaurant in the south", context)
    assert state == State.ASK_ADDITIONAL
    state, _, _ = manager.step(state, "no", context)
    assert state == State.SUGGEST
    state, _, _ = manager.step(state, "what is the phone number", context)
    assert state == State.GIVE_INFO
    state, _, _ = manager.step(state, "thank you", context)
    assert state == State.TERMINATE

def test_eager_suggests_after_first_preference() -> None:
    manager, context = make_manager(mode=Mode.EAGER), DialogContext()
    state, _, _ = manager.step(State.WELCOME, "i want italian food", context)
    assert state == State.SUGGEST
    assert context.restaurant["food"] == "italian"