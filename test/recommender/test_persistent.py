from pathlib import Path

from uu.msc.ai.mair.recommender.core.storage.experiment import (
    Experiment,
    Questionnarie,
    Session,
    Trial,
)
from uu.msc.ai.mair.recommender.core.storage.persistent import ExperimentData


def make_experiment(*sessions: Session) -> Experiment:
    return Experiment(
        participant_id="participant-1",
        age=25,
        gender="non-binary",
        sessions=list(sessions),
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

    assert data._load_experiment() == experiment


def test_update_experiment_appends_sessions_to_an_existing_experiment(tmp_path: Path) -> None:
    data = ExperimentData(tmp_path / "experiment.json")
    first_session = make_session("italian")
    second_session = make_session("thai")
    data.update_experiment(make_experiment(first_session))

    data.update_experiment(make_experiment(second_session))

    assert data._load_experiment().sessions == [first_session, second_session]
