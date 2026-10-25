"""Stores the experiment results the way the core storage does (``recommender.core.storage``).

Each completed system (three conversations plus the questionnaire) becomes one ``Experiment`` record, appended to
a local JSON file like ``ExperimentData.update_experiment()`` does; a participant who finishes the study has two
records that share a ``participant_id``. The records are extended here with what the web study adds (researcher,
person, AI usage, goals); the conversations themselves are not stored. The core code is unchanged.

Every researcher gets their own file, ``database-<researcher>.json`` in ``EXPERIMENT_DIR`` (by default the CLI's
``data/experiment/``, which git ignores), so the files can be merged later without clashing. Test runs go to
``test-<researcher>.json`` instead, so they never mix with the real data.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from uu.msc.ai.mair.recommender.core.storage.experiment import Experiment, Questionnarie, Session, Trial
from uu.msc.ai.mair.recommender.core.storage.persistent import ExperimentData

from chatbot import core

logger = logging.getLogger("chatbot.storage")

DEFAULT_DIR = core.REPO_ROOT / "data" / "experiment"


@dataclass(frozen=True)
class StudyTrial(Trial):
    """One conversation. ``food_type`` is the food of the last recommended restaurant ("" if none). The
    questionnaire is answered once per system, so only the last trial of a session carries the answers."""
    goal_slot: str = ""
    goal_value: str = ""
    restaurant: str | None = None
    started_at: datetime | None = None


@dataclass(frozen=True)
class StudySession(Session):
    trials: list[StudyTrial]
    label: str = ""
    order: int = 0


@dataclass(frozen=True)
class StudyExperiment(Experiment):
    session: StudySession
    researcher: str = ""
    person: int = 0
    group: str = ""
    ai_usage: int = 0
    test: bool = False
    completed_at: datetime | None = None


STUDY_ADAPTER = TypeAdapter(list[StudyExperiment])


class StudyData(ExperimentData):
    """``ExperimentData`` for the extended records (the core adapter would drop the extra fields)."""

    def _load_experiments(self) -> list[StudyExperiment]:
        return STUDY_ADAPTER.validate_json(self.experiment_path.read_text(encoding="utf-8"))

    def update_experiment(self, experiment: StudyExperiment) -> None:
        experiments = self._load_experiments() if self.experiment_path.exists() else []
        experiments.append(experiment)
        self.experiment_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_path.write_bytes(STUDY_ADAPTER.dump_json(experiments, indent=2))

    def load(self) -> list[StudyExperiment]:
        return self._load_experiments() if self.experiment_path.exists() else []


def directory() -> Path:
    return Path(os.environ.get("EXPERIMENT_DIR", DEFAULT_DIR))


def data(researcher: str, test: bool = False) -> StudyData:
    return StudyData(directory() / f"{'test' if test else 'database'}-{researcher.lower()}.json")


def save_experiment(experiment: StudyExperiment) -> None:
    study_data = data(experiment.researcher, experiment.test)
    study_data.update_experiment(experiment)
    logger.info("saved person %s, session %s (%s) to %s", experiment.person, experiment.session.order,
                experiment.session.mode, study_data.experiment_path)


def completed_sessions() -> dict[int, list[int]]:
    """For every person in the researchers' files on this computer, the sessions (1 and/or 2) that were completed."""
    done: dict[int, set[int]] = {}
    for path in sorted(directory().glob("database-*.json")):
        for experiment in StudyData(path).load():
            done.setdefault(experiment.person, set()).add(experiment.session.order)
    return {person: sorted(orders) for person, orders in done.items()}
