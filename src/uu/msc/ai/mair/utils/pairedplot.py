import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[6] / "data" / "experiment" / "scores.csv"
)
MEASURES = {"gaais_score": "GAAIS", "godspeed_score": "GSQ"}
MODES = ["eager", "patient"]


def plot_paired_scores(data: pd.DataFrame) -> None:
    """Plot the paired scores per measure and the distribution of their differences."""
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1.1]}
    )
    colors = {"eager": "tab:blue", "patient": "tab:orange"}
    # A fixed seed keeps the jitter, and therefore the figure, reproducible.
    rng = np.random.default_rng(0)

    # Draw one panel per measure, with one line per participant.
    differences = {}
    for ax, (score, title) in zip(axes, MEASURES.items()):
        paired = data.pivot_table(
            index="participant_id", columns="session.mode", values=score
        )[MODES]
        differences[title] = paired["patient"] - paired["eager"]

        # Connect the two scores of each participant, so the direction of change is visible.
        jitter = rng.uniform(-0.04, 0.04, size=len(paired))
        for (_, row), offset in zip(paired.iterrows(), jitter):
            ax.plot(
                [offset, 1 + offset],
                [row["eager"], row["patient"]],
                color="grey",
                alpha=0.4,
                linewidth=1,
            )

        # Add the individual scores and a box plot showing the median per mode.
        for position, mode in enumerate(MODES):
            ax.scatter(
                np.full(len(paired), position) + jitter,
                paired[mode],
                color=colors[mode],
                zorder=3,
                s=40,
            )
            ax.boxplot(
                paired[mode],
                positions=[position],
                widths=0.35,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"color": colors[mode]},
                whiskerprops={"color": colors[mode]},
                capprops={"color": colors[mode]},
            )
        test = wilcoxon(paired["patient"], paired["eager"])
        ax.set_xticks([0, 1], [mode.title() for mode in MODES])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0.8, 5.2)
        ax.set_ylabel(f"{title} score (1-5)")
        ax.set_title(f"{title}: W = {test.statistic:.1f}, p = {test.pvalue:.3f}")
        ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.hist(
        list(differences.values()),
        bins=np.arange(-3, 3.01, 0.4),
        label=list(differences),
        color=["tab:green", "tab:purple"],
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Patient - Eager")
    ax.set_ylabel("Participants")
    ax.set_title("Paired differences")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def main() -> None:
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    data = pd.read_csv(data_path)

    # Larger text, so the figure stays readable in the report.
    plt.rcParams.update({"font.size": 14})
    plot_paired_scores(data)
    plt.show()

if __name__ == "__main__":
    main()
