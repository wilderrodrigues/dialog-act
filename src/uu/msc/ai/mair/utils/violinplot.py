import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


# path to the csv 
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[6] / "data" / "experiment" / "full_database.csv"
)
# Questions 1–5 measure GAAIS; questions 6–10 measure GSQ.
GAAIS_QUESTIONS = {1, 2, 3, 4, 5}
GSQ_QUESTIONS = {6, 7, 8, 9, 10}
# Keep the mode order consistent in the plots and printed summaries.
MODES = ["eager", "patient"]


def plot_question_scores(
    data: pd.DataFrame,
    questions: set[int],
    title: str,
    reverse_questions: set[int] | None = None,
) -> None:
    """Plot answer distributions for selected questions, grouped by mode."""
    # Sorting ensures questions appear in numerical order on the x-axis.
    question_order = sorted(questions)
    # Keep only the selected questions and the two experiment modes.
    scores = data[
        data["question"].isin(questions) & data["session.mode"].isin(MODES)
    ].copy()

    # Reverse selected 1–5 Likert items so higher values consistently mean better scores.
    reverse_questions = reverse_questions or set()
    reverse_scored = scores["question"].isin(reverse_questions)
    scores.loc[reverse_scored, "answer"] = 6 - scores.loc[
        reverse_scored, "answer"
    ]

    # Print the number of answers in each question/mode group to show sample sizes.
    print(f"Number of scores per question and mode ({title}):")
    print(
        scores.groupby(["question", "session.mode"])["answer"]
        .count()
        .unstack(fill_value=0)
        .reindex(index=question_order, columns=MODES, fill_value=0)
        .to_string()
    )

    # Offset the mode distributions slightly so both violins are visible per question.
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"eager": "tab:blue", "patient": "tab:orange"}
    offsets = {"eager": -0.2, "patient": 0.2}

    # Draw one violin per question and mode using the individual raw answers.
    for question in question_order:
        for mode in MODES:
            values = scores.loc[
                (scores["question"] == question)
                & (scores["session.mode"] == mode),
                "answer",
            ].tolist()
            if not values:
                continue

            # Display the mode distributions side by side for this question.
            position = question + offsets[mode]
            violin = ax.violinplot(
                [values],
                positions=[position],
                widths=0.35,
                showmedians=True,
                showextrema=False,
            )
            violin["bodies"][0].set_facecolor(colors[mode])
            violin["bodies"][0].set_edgecolor(colors[mode])
            violin["bodies"][0].set_alpha(0.55)

            # Add jittered points so individual answers remain visible.
            jittered_positions = [
                position + ((index % 7) - 3) * 0.015
                for index in range(len(values))
            ]
            ax.scatter(jittered_positions, values, color=colors[mode], s=12, alpha=0.7)

    # Label the axes and mark every possible value on the 1–5 response scale.
    ax.set_xlabel(f"{title} question")
    ax.set_ylabel("Score (1–5)")
    ax.set_title(f"Scores per {title} question and mode")
    ax.set_xticks(question_order)
    ax.set_yticks(range(1, 6))
    ax.set_ylim(0.5, 5.5)
    ax.legend(
        handles=[Patch(facecolor=colors[mode], label=mode.title()) for mode in MODES],
        title="Mode",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def main() -> None:
    # Accept an optional CSV path; otherwise use the project's default data file.
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    # Detect the delimiter so comma- and semicolon-separated CSV files both work.
    data = pd.read_csv(data_path, sep=None, engine="python")

    # Convert the data columns to comparable numeric and lowercase values.
    data["question"] = pd.to_numeric(data["question"], errors="raise")
    data["answer"] = pd.to_numeric(data["answer"], errors="raise")
    data["session.mode"] = data["session.mode"].str.lower()

    # Reverse-score GAAIS questions 4 and 5; leave all GSQ answers unchanged.
    plot_question_scores(
        data,
        GAAIS_QUESTIONS,
        "GAAIS",
        reverse_questions={4, 5},
    )
    plot_question_scores(data, GSQ_QUESTIONS, "GSQ")
    plt.show()


# Run the plots only when this file is executed directly.
if __name__ == "__main__":
    main()
