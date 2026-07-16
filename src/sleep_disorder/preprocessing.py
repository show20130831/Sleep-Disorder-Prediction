"""Clean raw features into a model-ready DataFrame.

Every function here is a deterministic, split-safe transform (string parsing,
label mapping, category merging) — none of it "learns" anything from the data,
so it's safe to apply before the train/test split. One-hot encoding and
missing-value imputation are *not* here: those are stateful (they fit
parameters from data) and are deferred to ``models.build_preprocessor``, which
each model pipeline fits on its own training fold only, so no statistic
derived from validation/test data ever leaks into training.
"""

from __future__ import annotations

import pandas as pd

from sleep_disorder import config
from sleep_disorder.utils import get_logger

logger = get_logger(__name__)


def split_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Split ``"120/80"``-style Blood Pressure into Systolic_BP / Diastolic_BP."""
    df = df.copy()
    bp_parts = df[config.BLOOD_PRESSURE_COLUMN].str.split("/", expand=True)
    df[config.SYSTOLIC_BP_COLUMN] = pd.to_numeric(bp_parts[0], errors="coerce")
    df[config.DIASTOLIC_BP_COLUMN] = pd.to_numeric(bp_parts[1], errors="coerce")
    return df.drop(columns=[config.BLOOD_PRESSURE_COLUMN])


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw "Sleep Disorder" text label to the 3-class integer target."""
    df = df.copy()
    df[config.TARGET_COLUMN] = df[config.RAW_TARGET_COLUMN].map(
        config.SLEEP_DISORDER_LABEL_MAP
    )
    n_unmapped = df[config.TARGET_COLUMN].isna().sum()
    if n_unmapped:
        raise ValueError(
            f"{n_unmapped} rows have a '{config.RAW_TARGET_COLUMN}' value outside "
            f"{list(config.SLEEP_DISORDER_LABEL_MAP)}"
        )
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)
    return df


def merge_bmi_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate BMI labels (e.g. "Normal Weight" -> "Normal").

    The raw dataset encodes normal BMI as two distinct strings, which fragments
    one-hot encoding and dilutes statistical power.
    """
    df = df.copy()
    before = df["BMI Category"].value_counts().to_dict()
    df["BMI Category"] = df["BMI Category"].replace(config.BMI_CATEGORY_MERGE_MAP)
    logger.info("Merged BMI categories: %s", config.BMI_CATEGORY_MERGE_MAP)
    logger.debug(
        "BMI Category value counts %s -> %s", before, df["BMI Category"].value_counts().to_dict()
    )
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic, split-safe cleaning steps in sequence."""
    df = split_blood_pressure(df)
    df = encode_target(df)
    df = merge_bmi_categories(df)
    logger.info("Cleaned dataset: %d rows, %d columns", *df.shape)
    return df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the cleaned frame into model input ``X`` and target ``y``."""
    y = df[config.TARGET_COLUMN].astype(int)
    drop_cols = [config.ID_COLUMN, config.RAW_TARGET_COLUMN, config.TARGET_COLUMN]
    X = df.drop(columns=drop_cols)
    return X, y
