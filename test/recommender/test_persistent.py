from pathlib import Path

from uu.msc.ai.mair.recommender.core.storage.experiment import (
    Experiment,
    Questionnarie,
    Session,
    Trial,
)
from uu.msc.ai.mair.recommender.core.storage.persistent import ExperimentData


def make_experiment(session: Session) -> Experiment:
    return Experiment(
        age=25,
        gender="non-binary",
        session=session,
    )


def make_session(food_type: str) -> Session:
    return Session(
        mode="test",
        trials=[
            Trial(
                duration=30,
                nb_resets=1,
                nb_prompts=2,
                food_type=food_type,
                questionnaire=Questionnarie(answers=[(1, 2)]),
            )
        ],
    )


def test_update_experiment_creates_an_experiment_file(tmp_path: Path) -> None:
    experiment = make_experiment(make_session("italian"))
    data = ExperimentData(tmp_path / "experiment.json")

    data.update_experiment(experiment)

    assert data._load_experiments() == [experiment]


def test_update_experiment_appends_an_experiment_to_an_existing_file(tmp_path: Path) -> None:
    data = ExperimentData(tmp_path / "experiment.json")
    first_session = make_session("italian")
    second_session = make_session("thai")
    first_experiment = make_experiment(first_session)
    second_experiment = make_experiment(second_session)
    assert first_experiment.participant_id != second_experiment.participant_id

    data.update_experiment(first_experiment)
    data.update_experiment(second_experiment)

    experiments = data._load_experiments()
    assert experiments == [first_experiment, second_experiment]
