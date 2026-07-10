"""CLI entry point: preprocess -> descriptive analysis -> train -> evaluate -> feature importance.

Pure orchestration — each stage's logic lives in its own module. Run with:
    python -m sleep_disorder.main
or, after `pip install -e .`:
    sleep-disorder-pipeline
"""

from __future__ import annotations

from sleep_disorder import config
from sleep_disorder.data_loader import load_raw_data
from sleep_disorder.descriptive_analysis import run_descriptive_analysis
from sleep_disorder.evaluate import run_evaluation
from sleep_disorder.feature_importance import run_feature_importance
from sleep_disorder.preprocessing import clean_dataset, split_features_and_target
from sleep_disorder.train import split_train_test, train_all_models
from sleep_disorder.utils import configure_logging, get_logger, save_table

logger = get_logger(__name__)


def run_pipeline() -> None:
    """Run every pipeline stage end to end. Importable directly (e.g. from a notebook)."""
    logger.info("Starting sleep disorder prediction pipeline")

    raw_df = load_raw_data()
    logger.debug("Raw data shape: %s", raw_df.shape)
    cleaned_df = clean_dataset(raw_df)
    save_table(cleaned_df, config.PROCESSED_DATA_DIR / "sleep_cleaned.csv")

    X, y = split_features_and_target(cleaned_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    logger.debug("Train/test shapes: %s / %s", X_train.shape, X_test.shape)

    # Descriptive statistics run on the training split only: even though nothing
    # here feeds back into feature/model selection today (NUMERIC_FEATURES and
    # CATEGORICAL_FEATURES are fixed in config.py), keeping every code path off
    # the test set until final evaluation is the safer discipline to default to.
    run_descriptive_analysis(cleaned_df.loc[X_train.index])

    registry = train_all_models(X_train, y_train)
    summary = run_evaluation(
        registry, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    run_feature_importance(registry)

    logger.info("Pipeline complete. Model comparison:\n%s", summary.to_string(index=False))


def main() -> None:
    """CLI entry point: configure logging, run the pipeline, log-and-reraise on failure."""
    configure_logging()
    try:
        run_pipeline()
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
