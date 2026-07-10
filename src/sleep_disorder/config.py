"""Central configuration: paths, random seeds, column schemas, and model hyperparameters.

All magic numbers and hard-coded paths used across the pipeline live here so that
every stage (preprocessing, EDA, training, evaluation) shares a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "Sleep_health_and_lifestyle_dataset.csv"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
MODELS_DIR: Path = OUTPUTS_DIR / "models"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Target variable
# --------------------------------------------------------------------------- #
RAW_TARGET_COLUMN: str = "Sleep Disorder"
TARGET_COLUMN: str = "SleepDisorder_3class"

# 0 = None, 1 = Insomnia, 2 = Sleep Apnea
SLEEP_DISORDER_LABEL_MAP: dict[str, int] = {
    "None": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2,
}
CLASS_NAMES: list[str] = ["None", "Insomnia", "Sleep Apnea"]

# --------------------------------------------------------------------------- #
# Raw feature schema
# --------------------------------------------------------------------------- #
BLOOD_PRESSURE_COLUMN: str = "Blood Pressure"
SYSTOLIC_BP_COLUMN: str = "Systolic_BP"
DIASTOLIC_BP_COLUMN: str = "Diastolic_BP"

ID_COLUMN: str = "Person ID"

NUMERIC_FEATURES: list[str] = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    SYSTOLIC_BP_COLUMN,
    DIASTOLIC_BP_COLUMN,
]

CATEGORICAL_FEATURES: list[str] = ["Gender", "Occupation", "BMI Category"]

# Data-quality fix: the raw dataset encodes "normal BMI" as two distinct strings
# ("Normal" and "Normal Weight"), which fragments one-hot encoding and dilutes
# statistical power. Merge them before encoding. This changes downstream feature
# counts and model results relative to the original script (intentional; see README).
BMI_CATEGORY_MERGE_MAP: dict[str, str] = {
    "Normal Weight": "Normal",
}

# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.3
CV_FOLDS: int = 5

# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
# The training pipelines (SMOTE + optional class balancing) are designed to
# improve minority-class (Insomnia / Sleep Apnea) performance, so the primary
# comparison metric must weight all classes equally rather than favor the
# majority "None" class. Weighted F1 is retained as a secondary reference.
PRIMARY_SCORING: str = "f1_macro"
SECONDARY_SCORING: str = "f1_weighted"

TOP_N_FEATURES: int = 10

# Minimum expected cell frequency for the chi-square test of independence to be
# considered valid (Cochran's rule of thumb). Used to flag unreliable p-values
# for sparse categories (e.g., rare Occupation values).
CHI2_MIN_EXPECTED_FREQUENCY: float = 5.0


# --------------------------------------------------------------------------- #
# Model hyperparameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 100
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class XGBoostConfig:
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class LogisticRegressionConfig:
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class SmoteConfig:
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class PipelineConfig:
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    smote: SmoteConfig = field(default_factory=SmoteConfig)


PIPELINE_CONFIG = PipelineConfig()
