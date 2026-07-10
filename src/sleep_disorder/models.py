"""Model registry: builds the RF / XGBoost / MLR imbalanced-learn pipelines.

A single registry replaces the three near-duplicate training blocks in the
original script. Every pipeline shares the same preprocessing step, fit only
on the training fold (:func:`build_preprocessor`), and applies SMOTE only during ``fit``
(imblearn's ``Pipeline`` skips resamplers at predict time), so the held-out
fold is never oversampled. Hyperparameters come from ``config.PIPELINE_CONFIG``
and are unchanged from the original script, aside from dropping
``class_weight="balanced"``: with SMOTE already balancing each training fold
to equal class counts, that parameter was a no-op (see project README for the
derivation) and removing it does not change results.
"""

from __future__ import annotations

from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from sleep_disorder import config
from sleep_disorder.config import PIPELINE_CONFIG


RANDOM_FOREST_KEY = "random_forest"
XGBOOST_KEY = "xgboost"
LOGISTIC_REGRESSION_KEY = "logistic_regression"
MODEL_KEYS = (RANDOM_FOREST_KEY, XGBOOST_KEY, LOGISTIC_REGRESSION_KEY)


@dataclass(frozen=True)
class ModelSpec:
    """A named model pipeline plus the step name of its final estimator."""

    display_name: str
    pipeline: ImbPipeline
    estimator_step: str


def build_preprocessor() -> ColumnTransformer:
    """Build the encoding/imputation step shared by every model pipeline.

    - Numeric features: mean-imputed (defensive; this dataset has no missing
      values, but imputing here rather than in ``preprocessing.py`` means the
      mean is always computed from the training fold only).
    - Categorical features: one-hot encoded with the first category dropped,
      matching ``pd.get_dummies(..., drop_first=True)`` from the original
      script. ``handle_unknown="ignore"`` makes it robust to a category that
      happens to appear only in the held-out fold.

    Fitting this inside each model's :class:`~imblearn.pipeline.Pipeline` means
    it is refit on every train fold independently — no statistic derived from
    validation/test data ever leaks into training.
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_random_forest_pipeline() -> ImbPipeline:
    cfg = PIPELINE_CONFIG.random_forest
    return ImbPipeline(
        [
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=PIPELINE_CONFIG.smote.random_state)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=cfg.n_estimators, random_state=cfg.random_state
                ),
            ),
        ]
    )


def build_xgboost_pipeline() -> ImbPipeline:
    cfg = PIPELINE_CONFIG.xgboost
    return ImbPipeline(
        [
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=PIPELINE_CONFIG.smote.random_state)),
            (
                "xgb",
                XGBClassifier(
                    objective=cfg.objective,
                    eval_metric=cfg.eval_metric,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def build_logistic_regression_pipeline() -> ImbPipeline:
    cfg = PIPELINE_CONFIG.logistic_regression
    return ImbPipeline(
        [
            ("preprocess", build_preprocessor()),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=PIPELINE_CONFIG.smote.random_state)),
            (
                "mlr",
                LogisticRegression(
                    solver=cfg.solver, max_iter=cfg.max_iter, random_state=cfg.random_state
                ),
            ),
        ]
    )


def get_model_registry() -> dict[str, ModelSpec]:
    """Build a fresh, unfitted registry of all three model pipelines."""
    return {
        RANDOM_FOREST_KEY: ModelSpec(
            "Random Forest", build_random_forest_pipeline(), estimator_step="rf"
        ),
        XGBOOST_KEY: ModelSpec("XGBoost", build_xgboost_pipeline(), estimator_step="xgb"),
        LOGISTIC_REGRESSION_KEY: ModelSpec(
            "Multinomial Logistic Regression",
            build_logistic_regression_pipeline(),
            estimator_step="mlr",
        ),
    }
