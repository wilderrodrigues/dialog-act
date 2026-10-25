from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from uu.msc.ai.mair.recommender.core.storage.experiment import Experiment


class ExperimentData:

    def __init__(self, experiment_path: Path) -> None:
        self.experiment_path = experiment_path


    def _load_experiment(self) -> Experiment:
        return TypeAdapter(Experiment).validate_json(
            self.experiment_path.read_text(encoding="utf-8")
        )

    def update_experiment(self, experiment: Experiment) -> None:
        if self.experiment_path.exists():
            stored_experiment = self._load_experiment()
            experiment = replace(
                stored_experiment,
                sessions=[*stored_experiment.sessions, *experiment.sessions],
            )

        self.experiment_path.write_bytes(
            TypeAdapter(Experiment).dump_json(experiment, indent=2)
        )
