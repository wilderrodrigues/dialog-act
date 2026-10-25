"""Merges the database.json files of the researchers into one, in the same ``ExperimentData`` format.

Every researcher runs the study on their own laptop, so there is one file per researcher. Records that occur in
more than one file (same participant and session) are kept once.

    python scripts/merge_results.py merged.json database-remy.json database-cai.json database-dexter.json database-wilder.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.storage import STUDY_ADAPTER, StudyData  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    output, *inputs = map(Path, sys.argv[1:])
    experiments = {}
    for path in inputs:
        for experiment in StudyData(path).load():
            experiments.setdefault((experiment.participant_id, experiment.session.order), experiment)
    merged = sorted(experiments.values(), key=lambda experiment: (experiment.person, experiment.session.order))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(STUDY_ADAPTER.dump_json(merged, indent=2))
    print(f"Wrote {len(merged)} records from {len({e.participant_id for e in merged})} participants to {output}")


if __name__ == "__main__":
    main()
