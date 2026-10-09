from uu import get_root
from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory
from uu.msc.ai.mair.dialog.model.rule_based_classifier import RuleBasedModel


def test_rule_based_predicts_keyword_acts() -> None:
    model = RuleBasedModel()
    assert model.predict_one("thank you good bye") == "thankyou"
    assert model.predict_one("yes") == "affirm"
    assert model.predict_one("what is the phone number") == "request"
    assert model.predict_one("how about italian food") == "reqalts"


def test_rule_based_falls_back_to_the_default_act() -> None:
    assert RuleBasedModel().predict_one("cheap food") == RuleBasedModel.DEFAULT_ACT


def test_rule_based_first_matching_rule_wins() -> None:
    assert RuleBasedModel().predict_one("no thank you") == "thankyou"


def test_rule_based_predicts_every_utterance() -> None:
    dataset_file = get_root() / "test" / "resources" / "test_acts.dat"
    utterances, acts, targets = DatasetFactory.load_test_dataset(data_path=dataset_file)

    predictions = RuleBasedModel().predict(utterances)
    assert predictions.shape == acts.shape
    assert set(predictions) <= set(targets) | set(RuleBasedModel.KEYWORDS)
