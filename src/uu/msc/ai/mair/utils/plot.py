import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import pearsonr


gaais_col = "gaais_score"
gsq_col = "godspeed_score"

gaais_questions = ["q1", "q2", "q3", "q4", "q5"]
gsq_questions = ["q6", "q7", "q8", "q9", "q10"]

def means(df: pd.DataFrame) -> None:
    # Make comparisons robust to Patient/patient, Female/female, etc.
    df["session.mode"] = df["session.mode"].str.lower()
    df["gender"] = df["gender"].str.lower()

    question_means = df[gaais_questions + gsq_questions].mean()

    # Add a gap between GAAIS and GSQ
    x_gaais = np.arange(5)
    x_gsq = np.arange(6, 11)

    plt.figure(figsize=(10, 5))

    plt.bar(
        x_gaais,
        question_means[gaais_questions],
        label="GAAIS"
    )

    plt.bar(
        x_gsq,
        question_means[gsq_questions],
        label="GSQ"
    )

    plt.xticks(
        np.concatenate([x_gaais, x_gsq]),
        gaais_questions + gsq_questions
    )

    plt.xlabel("Question")
    plt.ylabel("Mean score")
    plt.title("Mean Scores per GAAIS and GSQ Question")
    plt.ylim(0, 5.5)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def modes(df: pd.DataFrame) -> None:
    treatment_patient = df[
        (df["session.mode"] == "patient")
        ]

    treatment_eager = df[
        (df["session.mode"] == "eager")
        ]

    treatment_patient_means = [
        treatment_patient[gaais_col].mean(),
        treatment_patient[gsq_col].mean()
    ]

    treatment_eager_means = [
        treatment_eager[gaais_col].mean(),
        treatment_eager[gsq_col].mean()
    ]

    labels = ["GAAIS", "GSQ"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        x - width / 2,
        treatment_patient_means,
        width,
        label="Patient"
    )

    plt.bar(
        x + width / 2,
        treatment_eager_means,
        width,
        label="Eager"
    )

    plt.xticks(x, labels)
    plt.ylabel("Mean score")
    plt.title("GAAIS and GSQ by Treatment")
    plt.ylim(0, 5.5)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def genres(df: pd.DataFrame) -> None:
    female = df[df["gender"] == "female"]
    male = df[df["gender"] == "male"]

    female_means = [
        female[gaais_col].mean(),
        female[gsq_col].mean()
    ]

    male_means = [
        male[gaais_col].mean(),
        male[gsq_col].mean()
    ]

    labels = ["GAAIS", "GSQ"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        x - width / 2,
        female_means,
        width,
        label="Female"
    )

    plt.bar(
        x + width / 2,
        male_means,
        width,
        label="Male"
    )

    plt.xticks(x, labels)
    plt.ylabel("Mean score")
    plt.title("GAAIS and GSQ by Gender")
    plt.ylim(0, 5.5)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def density(df: pd.DataFrame, score: str="gaais_score", title: str="GAAIS Scores") -> None:
    plt.figure(figsize=(9, 5))

    for system in ["patient", "eager"]:
        values = df.loc[
            df["session.mode"] == system,
            score
        ].dropna()

        mu, sigma = norm.fit(values)

        x = np.linspace(1, 5, 300)
        y = norm.pdf(x, mu, sigma)

        plt.plot(
            x,
            y,
            linewidth=2,
            label=f"{system} (μ={mu:.2f}, σ={sigma:.2f})"
        )

    plt.xlabel(f"Questionnaire {title}")
    plt.ylabel("Probability density")
    plt.title(f"Distribution of Questionnaire {title}")

    plt.xlim(1, 5)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def ai_usage_correlation(df: pd.DataFrame, score: str="gaais_score", title: str="GAAIS Scores") -> None:

    plt.figure(figsize=(8, 5))

    for system in ["patient", "eager"]:

        subset = (
            df[df["session.mode"] == system]
            [["ai_usage", score]]
            .dropna()
        )

        x = subset["ai_usage"].astype(float).to_numpy()
        y = subset[score].astype(float).to_numpy()

        # Scatter
        plt.scatter(
            x,
            y,
            alpha=0.7,
            label=system
        )

        # Linear regression line
        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_line = slope * x_line + intercept

        plt.plot(
            x_line,
            y_line,
            linewidth=2
        )

        # Pearson correlation
        r, p = pearsonr(x, y)

        print(
            f"{title} — {system}: "
            f"r={r:.3f}, p={p:.4f}"
        )

    plt.xlabel("AI Usage")
    plt.ylabel(f"{title}")
    plt.title(f"{title} vs AI Usage")

    plt.xticks(sorted(df["ai_usage"].dropna().unique()))

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()