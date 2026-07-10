"""Evaluate trained pipelines: classification report, 5-fold CV, One-vs-Rest ROC/AUC.

Reports both macro F1 and weighted F1 for every metric, but treats macro F1 as
the primary comparison metric. The training pipelines apply SMOTE specifically
to lift minority-class (Insomnia / Sleep Apnea) performance; weighted F1 is
dominated by the majority "None" class (58.6% of the data) and can mask
whether that effort actually paid off, so it is kept only as a secondary,
familiar reference point.
"""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import label_binarize

from sleep_disorder import config
from sleep_disorder.models import ModelSpec
from sleep_disorder.utils import get_logger, save_figure, save_json, save_table

logger = get_logger(__name__)

_ROC_COLORS = ("aqua", "darkorange", "cornflowerblue")


def evaluate_on_test(
    pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    """Predict on the held-out test set and return a classification report dict.

    The dict includes per-class precision/recall/f1, plus sklearn's built-in
    ``"macro avg"`` and ``"weighted avg"`` keys.
    """
    y_pred = pipeline.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=config.CLASS_NAMES, output_dict=True
    )
    logger.info(
        "Test macro F1=%.4f, weighted F1=%.4f",
        report["macro avg"]["f1-score"],
        report["weighted avg"]["f1-score"],
    )
    return report


def run_cross_validation(
    pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Any]:
    """5-fold CV, scored on both macro and weighted F1 in a single pass."""
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=config.CV_FOLDS,
        scoring=[config.PRIMARY_SCORING, config.SECONDARY_SCORING],
        n_jobs=-1,
    )
    summary = {
        f"cv_{config.PRIMARY_SCORING}_mean": float(np.mean(cv_results[f"test_{config.PRIMARY_SCORING}"])),
        f"cv_{config.PRIMARY_SCORING}_std": float(np.std(cv_results[f"test_{config.PRIMARY_SCORING}"])),
        f"cv_{config.SECONDARY_SCORING}_mean": float(
            np.mean(cv_results[f"test_{config.SECONDARY_SCORING}"])
        ),
        f"cv_{config.SECONDARY_SCORING}_std": float(
            np.std(cv_results[f"test_{config.SECONDARY_SCORING}"])
        ),
    }
    logger.info(
        "CV %s: %.4f (+/- %.4f) | CV %s: %.4f (+/- %.4f)",
        config.PRIMARY_SCORING,
        summary[f"cv_{config.PRIMARY_SCORING}_mean"],
        summary[f"cv_{config.PRIMARY_SCORING}_std"],
        config.SECONDARY_SCORING,
        summary[f"cv_{config.SECONDARY_SCORING}_mean"],
        summary[f"cv_{config.SECONDARY_SCORING}_std"],
    )
    return summary


def compute_roc_auc(y_test: pd.Series, y_proba: np.ndarray) -> dict[str, Any]:
    """One-vs-Rest ROC curves/AUC per class, plus the macro-average AUC."""
    y_test_bin = label_binarize(y_test, classes=list(range(len(config.CLASS_NAMES))))
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        if np.all(y_proba[:, i] == 0) or np.all(y_proba[:, i] == 1):
            fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), np.nan
        else:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    macro_auc = float(np.nanmean(list(roc_auc.values())))
    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "macro_auc": macro_auc}


def _plot_roc_subplot(ax: plt.Axes, roc_result: dict, model_name: str) -> None:
    colors = cycle(_ROC_COLORS)
    for i, color in zip(range(len(config.CLASS_NAMES)), colors):
        class_auc = roc_result["roc_auc"][i]
        label = (
            f"{config.CLASS_NAMES[i]} (AUC = {class_auc:.2f})"
            if not np.isnan(class_auc)
            else f"{config.CLASS_NAMES[i]} (AUC N/A)"
        )
        ax.plot(roc_result["fpr"][i], roc_result["tpr"][i], color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC curve")
    ax.legend(loc="lower right")


def plot_roc_curves(roc_results: dict[str, dict], output_path: Path | None = None) -> None:
    """Save a combined 1x3 ROC figure, one subplot per model."""
    output_path = output_path or config.FIGURES_DIR / "roc_curves.png"
    fig, axes = plt.subplots(1, len(roc_results), figsize=(6 * len(roc_results), 6))
    for ax, (model_name, roc_result) in zip(np.atleast_1d(axes), roc_results.items()):
        _plot_roc_subplot(ax, roc_result, model_name)
    save_figure(fig, output_path)
    logger.info("Saved ROC curves to %s", output_path)


def run_evaluation(
    registry: dict[str, ModelSpec],
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate every model in the registry and save reports/figures/summary table.

    ``X_train``/``y_train`` (fit the 5-fold CV) and ``X_test``/``y_test`` (final,
    read-only prediction only) are keyword-only on purpose: four same-typed
    DataFrame/Series positional arguments in a row is an easy way to
    accidentally swap train and test at the call site with no error from
    Python or mypy. Keyword-only arguments force every call site to name which
    is which.
    """
    roc_results: dict[str, dict] = {}
    summary_rows = []

    for key, spec in registry.items():
        report = evaluate_on_test(spec.pipeline, X_test, y_test)
        save_json(report, config.TABLES_DIR / f"classification_report_{key}.json")

        cv_summary = run_cross_validation(spec.pipeline, X_train, y_train)

        y_proba = spec.pipeline.predict_proba(X_test)
        roc_result = compute_roc_auc(y_test, y_proba)
        roc_results[spec.display_name] = roc_result

        summary_rows.append(
            {
                "model": spec.display_name,
                "test_f1_macro": report["macro avg"]["f1-score"],
                "test_f1_weighted": report["weighted avg"]["f1-score"],
                "test_accuracy": report["accuracy"],
                "macro_auc": roc_result["macro_auc"],
                **cv_summary,
            }
        )

    plot_roc_curves(roc_results)

    summary = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)
    save_table(summary, config.TABLES_DIR / "model_comparison_summary.csv")
    logger.info("Saved model_comparison_summary.csv")
    return summary
