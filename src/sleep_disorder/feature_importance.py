"""Extract and plot feature importance (RF, XGBoost) and coefficients (MLR).

Also computes the cross-model top-feature intersection: the features that
rank in every model's top N, ranked by Random Forest importance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline

from sleep_disorder import config
from sleep_disorder.models import (
    LOGISTIC_REGRESSION_KEY,
    RANDOM_FOREST_KEY,
    XGBOOST_KEY,
    ModelSpec,
)
from sleep_disorder.utils import get_logger, save_figure, save_json, save_table

logger = get_logger(__name__)


def _clean_feature_name(name: str) -> str:
    """Strip the ColumnTransformer's "num__"/"cat__" prefix for readable plots."""
    return name.split("__", 1)[-1]


def get_transformed_feature_names(pipeline: ImbPipeline) -> list[str]:
    raw_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [_clean_feature_name(n) for n in raw_names]


def compute_tree_importance(pipeline: ImbPipeline, estimator_step: str) -> pd.Series:
    """Full, descending-sorted feature importance series for a tree-based model."""
    feature_names = get_transformed_feature_names(pipeline)
    estimator = pipeline.named_steps[estimator_step]
    return pd.Series(estimator.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


def compute_mlr_coefficients(pipeline: ImbPipeline) -> pd.DataFrame:
    """Per-class coefficients (features x classes) plus a Mean_Abs_Coef column."""
    feature_names = get_transformed_feature_names(pipeline)
    estimator = pipeline.named_steps["mlr"]
    coef_df = pd.DataFrame(
        estimator.coef_.T,
        index=feature_names,
        columns=[f"vs {name}" for name in config.CLASS_NAMES[: estimator.coef_.shape[0]]],
    )
    coef_df["Mean_Abs_Coef"] = coef_df.abs().mean(axis=1)
    return coef_df.sort_values("Mean_Abs_Coef", ascending=False)


def plot_tree_importances(
    rf_importance: pd.Series, xgb_importance: pd.Series, output_path: Path | None = None
) -> None:
    output_path = output_path or config.FIGURES_DIR / "feature_importance_tree_models.png"
    top_n = config.TOP_N_FEATURES

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rf_top = rf_importance.head(top_n)
    sns.barplot(x=rf_top.values, y=rf_top.index, hue=rf_top.index, legend=False,
                palette="viridis", ax=axes[0])
    axes[0].set_title("Random Forest Feature Importance")
    axes[0].set_xlabel("Importance Score")
    axes[0].set_ylabel("Features")

    xgb_top = xgb_importance.head(top_n)
    sns.barplot(x=xgb_top.values, y=xgb_top.index, hue=xgb_top.index, legend=False,
                palette="mako", ax=axes[1])
    axes[1].set_title("XGBoost Feature Importance")
    axes[1].set_xlabel("Importance Score")
    axes[1].set_ylabel("Features")

    save_figure(fig, output_path)
    logger.info("Saved tree-model feature importance plot to %s", output_path)


def plot_mlr_heatmap(mlr_coefficients: pd.DataFrame, output_path: Path | None = None) -> None:
    output_path = output_path or config.FIGURES_DIR / "mlr_coefficient_heatmap.png"
    top_coefs = mlr_coefficients.head(config.TOP_N_FEATURES).drop(columns=["Mean_Abs_Coef"])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        top_coefs,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        center=0,
        cbar_kws={"label": "Coefficient Value (Multinomial)"},
        ax=ax,
    )
    ax.set_title("Multinomial Logistic Regression Top 10 Absolute Coefficients")
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Standardized Features")

    save_figure(fig, output_path)
    logger.info("Saved MLR coefficient heatmap to %s", output_path)


def compute_cross_model_top_features(
    rf_importance: pd.Series, xgb_importance: pd.Series, mlr_coefficients: pd.DataFrame
) -> list[str]:
    """Features that land in every model's top-N list, ranked by RF importance."""
    top_n = config.TOP_N_FEATURES
    rf_top = set(rf_importance.head(top_n).index)
    xgb_top = set(xgb_importance.head(top_n).index)
    mlr_top = set(mlr_coefficients.head(top_n).index)

    intersection = rf_top & xgb_top & mlr_top
    ranked = [f for f in rf_importance.index if f in intersection]
    logger.info("Top-%d features common to all three models: %s", top_n, ranked)
    return ranked


def run_feature_importance(registry: dict[str, ModelSpec]) -> dict[str, Any]:
    """Compute, plot, and save feature importance for all three models."""
    rf_importance = compute_tree_importance(
        registry[RANDOM_FOREST_KEY].pipeline, registry[RANDOM_FOREST_KEY].estimator_step
    )
    xgb_importance = compute_tree_importance(
        registry[XGBOOST_KEY].pipeline, registry[XGBOOST_KEY].estimator_step
    )
    mlr_coefficients = compute_mlr_coefficients(registry[LOGISTIC_REGRESSION_KEY].pipeline)

    save_table(
        rf_importance.rename("importance").rename_axis("feature").reset_index(),
        config.TABLES_DIR / "feature_importance_random_forest.csv",
    )
    save_table(
        xgb_importance.rename("importance").rename_axis("feature").reset_index(),
        config.TABLES_DIR / "feature_importance_xgboost.csv",
    )
    save_table(
        mlr_coefficients.rename_axis("feature").reset_index(),
        config.TABLES_DIR / "mlr_coefficients.csv",
    )

    plot_tree_importances(rf_importance, xgb_importance)
    plot_mlr_heatmap(mlr_coefficients)

    common_features = compute_cross_model_top_features(
        rf_importance, xgb_importance, mlr_coefficients
    )
    save_json(
        {"top_n": config.TOP_N_FEATURES, "common_features": common_features},
        config.TABLES_DIR / "cross_model_top_features.json",
    )

    return {
        "random_forest": rf_importance,
        "xgboost": xgb_importance,
        "logistic_regression": mlr_coefficients,
        "common_features": common_features,
    }
