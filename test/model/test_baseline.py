from uu.msc.ai.mair.dialog.core.splits import get_split
from uu.msc.ai.mair.dialog.metrics.evaluation import compute_metrics
from uu.msc.ai.mair.dialog.model.baseline import DEFAULT_ACT, predict, predict_one


def test_rule_order_resolves_overlapping_keywords() -> None:
    assert predict_one("thank you good bye") == "thankyou"
    assert predict_one("no thank you good bye") == "negate"
    assert predict_one("okay good bye") == "bye"
    assert predict_one("any price range") == "inform"
    assert predict_one("what is the price range") == "request"


def test_examples_from_assignment_table() -> None:
    examples = {
        "okay um": "ack",
        "yes right": "affirm",
        "see you good bye": "bye",
        "is it in the center of town": "confirm",
        "i dont want vietnamese food": "deny",
        "hi i want a restaurant": "hello",
        "im looking for a restaurant that serves seafood": "inform",
        "no in any area": "negate",
        "cough": "null",
        "can you repeat that": "repeat",
        "how about korean food": "reqalts",
        "more": "reqmore",
        "what is the post code": "request",
        "okay start over": "restart",
        "thank you goodbye": "thankyou",
    }
    assert predict(list(examples)) == list(examples.values())


def test_input_is_lowercased_and_unknown_falls_back_to_default() -> None:
    assert predict_one("THANK YOU") == "thankyou"
    assert predict_one("xyzzy") == DEFAULT_ACT
    assert predict_one("") == DEFAULT_ACT


def test_baseline_reaches_required_accuracy_on_both_splits() -> None:
    for split_name in ("original", "grouped"):
        _, x_test, _, y_test = get_split(split_name)
        assert compute_metrics(y_test, predict(x_test))["accuracy"] >= 0.80
