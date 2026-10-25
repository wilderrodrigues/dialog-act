from pathlib import Path

from pydantic import TypeAdapter

from uu.msc.ai.mair.recommender.core.storage.experiment import Experiment


EXPERIMENTS_ADAPTER = TypeAdapter(list[Experiment])


class ExperimentData:

    def __init__(self, experiment_path: Path) -> None:
        self.experiment_path = experiment_path


    def _load_experiments(self) -> list[Experiment]:
        return EXPERIMENTS_ADAPTER.validate_json(
            self.experiment_path.read_text(encoding="utf-8")
        )

    def update_experiment(self, experiment: Experiment) -> None:
        experiments = self._load_experiments() if self.experiment_path.exists() else []
        experiments.append(experiment)

        self.experiment_path.write_bytes(
            EXPERIMENTS_ADAPTER.dump_json(experiments, indent=2)
        )
