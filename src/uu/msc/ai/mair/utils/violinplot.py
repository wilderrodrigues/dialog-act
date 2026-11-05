import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Default input file; a different CSV can be supplied as a command-line argument.
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[6] / "data" / "experiment" / "full_database.csv"
)
# These are the question IDs used by each questionnaire in the source CSV.
QUESTIONNAIRES = {
    "GAAIS": (1, 2, 3, 4, 5),
    "GSQ": (6, 7, 8, 9, 10),
}


def calculate_participant_differences(data: pd.DataFrame) -> pd.DataFrame:
    """Return one patient-minus-eager mean difference per questionnaire and participant."""
    # Keep only survey answers used in the two questionnaires and the two relevant modes.
    questions = [question for group in QUESTIONNAIRES.values() for question in group]
    data = data.loc[
        data["question"].isin(questions)
        & data["session.mode"].str.lower().isin(["eager", "patient"]),
        ["participant_id", "question", "answer", "session.mode"],
    ].copy()

    # Convert the IDs and scores to numbers, and match mode names regardless of case.
    data["question"] = pd.to_numeric(data["question"], errors="raise")
    data["answer"] = pd.to_numeric(data["answer"], errors="raise")
    data["session.mode"] = data["session.mode"].str.lower()

    # Reverse-score GAAIS items 4 and 5 on the 1–5 scale before comparing the modes.
    reversed_items = data["question"].isin([4, 5])
    data.loc[reversed_items, "answer"] = 6 - data.loc[reversed_items, "answer"]

    # Place both mode answers side by side for each participant and question.
    answers = data.pivot(
        index=["participant_id", "question"], columns="session.mode", values="answer"
    ).reindex(columns=["eager", "patient"])

    # Subtract eager from patient for each question, yielding one row per participant.
    differences = (answers["patient"] - answers["eager"]).unstack("question").reindex(
        columns=questions
    )

    # Average the five question differences; require all five paired answers per scale.
    return pd.DataFrame(
        {
            name: differences[list(items)].mean(axis=1).where(
                differences[list(items)].notna().all(axis=1)
            )
            for name, items in QUESTIONNAIRES.items()
        }
    ).dropna()


def plot_participant_differences(scores: pd.DataFrame) -> None:
    """Draw one violin per questionnaire from the participant-level differences."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"GAAIS": "tab:blue", "GSQ": "tab:orange"}

    # Keep the x-axis order consistent with QUESTIONNAIRES: GAAIS, then GSQ.
    names = list(QUESTIONNAIRES)
    violins = ax.violinplot(
        # Each list holds one mean difference per participant, not individual answers.
        [scores[name].tolist() for name in names],
        positions=[1, 2],
        widths=0.65,
        showmedians=True,
        showextrema=False,
    )
    # Color each violin to distinguish the two questionnaires.
    for body, name in zip(violins["bodies"], names):
        body.set_facecolor(colors[name])
        body.set_edgecolor(colors[name])
        body.set_alpha(0.55)

    # Show each participant's score as a dot; slight horizontal offsets prevent overlap.
    for position, name in enumerate(names, start=1):
        values = scores[name].tolist()
        offsets = [position + ((index % 7) - 3) * 0.025 for index in range(len(values))]
        ax.scatter(offsets, values, color=colors[name], s=22, alpha=0.8, zorder=3)

    # Zero means no difference between modes; positive values mean patient scored higher.
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([1, 2], names)
    ax.set_ylabel("Mean score difference")
    ax.set_title("GAAIS and GSQ score differences per participant\n(patient − eager)")
    # With a 1–5 response scale, the difference range is -4 to +4.
    ax.set_yticks(range(-4, 5))
    ax.set_ylim(-4.2, 4.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def main() -> None:
    # Use a provided CSV path, or fall back to the project's experiment data.
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    # Detect the delimiter so comma- and semicolon-separated CSVs both work.
    data = pd.read_csv(data_path, sep=None, engine="python")
    participant_scores = calculate_participant_differences(data)
    if participant_scores.empty:
        raise ValueError("No participants have complete paired answers for both questionnaires.")

    plot_participant_differences(participant_scores)
    plt.show()


if __name__ == "__main__":
    main()
