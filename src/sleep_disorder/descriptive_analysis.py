"""Descriptive statistics: ANOVA for numeric features, chi-square for categorical features.

Every function here is agnostic to *which* rows it's given — the caller
decides. ``main.run_pipeline`` passes only the training split, keeping every
code path off the test set until final evaluation.

Categorical variables are tested one-hot dummy column vs. target. Rare
categories (e.g. an occupation held by only a handful of people) can violate
the chi-square test's minimum expected cell frequency assumption;
:func:`compute_categorical_summary` flags this per column instead of silently
reporting an unreliable p-value.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

from sleep_disorder import config
from sleep_disorder.utils import get_logger, save_figure, save_table

logger = get_logger(__name__)


def compute_numeric_summary(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    target_col: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """Per-class mean/SD and one-way ANOVA p-value for each numeric feature."""
    numeric_cols = numeric_cols if numeric_cols is not None else config.NUMERIC_FEATURES
    rows = []
    groups = [df.loc[df[target_col] == cls, :] for cls in range(len(config.CLASS_NAMES))]

    for col in numeric_cols:
        group_values = [g[col] for g in groups]
        _, p_value = f_oneway(*group_values)

        row: dict[str, str | float] = {"Variable": col}
        for cls, g in zip(config.CLASS_NAMES, group_values):
            row[f"Mean_{cls}"] = g.mean()
            row[f"SD_{cls}"] = g.std()
        row["ANOVA_p_value"] = p_value
        rows.append(row)

    return pd.DataFrame(rows)


def plot_numeric_boxplots(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_col: str,
    output_dir: Path | None = None,
) -> None:
    """Save one boxplot per numeric feature, grouped by sleep disorder class."""
    output_dir = output_dir if output_dir is not None else config.FIGURES_DIR
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[target_col], y=df[col], hue=df[target_col], legend=False, ax=ax)
        ax.set_title(f"{col} across Sleep Disorder Groups")
        ax.set_xlabel("Sleep Disorder Group (0=None, 1=Insomnia, 2=Apnea)")
        ax.set_ylabel(col)
        save_figure(fig, output_dir / f"boxplot_{col.replace(' ', '_')}.png")
    logger.info("Saved %d boxplots to %s", len(numeric_cols), output_dir)


def one_hot_encode_for_eda(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns for EDA only (not used by models)."""
    df_encoded = pd.get_dummies(df, columns=config.CATEGORICAL_FEATURES, drop_first=True)
    categorical_dummy_cols = [
        col
        for col in df_encoded.columns
        if any(col.startswith(f"{c}_") for c in config.CATEGORICAL_FEATURES)
    ]
    return df_encoded, categorical_dummy_cols


def compute_categorical_summary(
    df_encoded: pd.DataFrame,
    categorical_dummy_cols: list[str],
    target_col: str = config.TARGET_COLUMN,
) -> pd.DataFrame:
    """Chi-square test of independence for each one-hot dummy column vs. target.

    Flags columns whose contingency table has any expected cell frequency below
    ``config.CHI2_MIN_EXPECTED_FREQUENCY`` — the p-value for those rows should be
    treated as unreliable rather than reported at face value.
    """
    rows = []
    for col in categorical_dummy_cols:
        contingency = pd.crosstab(df_encoded[col], df_encoded[target_col])
        chi2, p_value, _, expected = chi2_contingency(contingency)
        assumption_violated = bool((expected < config.CHI2_MIN_EXPECTED_FREQUENCY).any())
        rows.append(
            {
                "Variable": col,
                "Chi-square p-value": p_value,
                "Assumption_violated (expected count < 5)": assumption_violated,
            }
        )

    result = pd.DataFrame(rows)
    n_violated = result["Assumption_violated (expected count < 5)"].sum()
    if n_violated:
        logger.warning(
            "%d/%d categorical variables violate the chi-square minimum expected "
            "frequency assumption; treat their p-values with caution.",
            n_violated,
            len(result),
        )
    return result


def run_descriptive_analysis(df_cleaned: pd.DataFrame) -> None:
    """Run the full descriptive analysis and save tables/figures to outputs/."""
    numeric_summary = compute_numeric_summary(df_cleaned)
    save_table(numeric_summary, config.TABLES_DIR / "descriptive_numeric_results.csv")
    logger.info("Saved descriptive_numeric_results.csv")

    plot_numeric_boxplots(df_cleaned, config.NUMERIC_FEATURES, config.TARGET_COLUMN)

    df_encoded, categorical_dummy_cols = one_hot_encode_for_eda(df_cleaned)
    categorical_summary = compute_categorical_summary(df_encoded, categorical_dummy_cols)
    save_table(categorical_summary, config.TABLES_DIR / "descriptive_categorical_results.csv")
    logger.info("Saved descriptive_categorical_results.csv")
