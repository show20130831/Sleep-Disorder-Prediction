"""Split the data, fit each registered model pipeline, and persist it to disk."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from sleep_disorder import config
from sleep_disorder.models import ModelSpec, get_model_registry
from sleep_disorder.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 70/30 train/test split (matches the original script)."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, ModelSpec]:
    """Fit every model in the registry on the training split and save it to disk."""
    registry = get_model_registry()
    ensure_dir(config.MODELS_DIR)

    for key, spec in registry.items():
        logger.info("Training %s ...", spec.display_name)
        spec.pipeline.fit(X_train, y_train)

        model_path = config.MODELS_DIR / f"{key}.joblib"
        joblib.dump(spec.pipeline, model_path)
        logger.info("Saved %s to %s", spec.display_name, model_path)

    return registry
