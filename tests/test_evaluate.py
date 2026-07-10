import numpy as np
import pandas as pd

from sleep_disorder.evaluate import compute_roc_auc


def test_compute_roc_auc_perfect_predictions_give_auc_one():
    y_test = pd.Series([0, 0, 1, 1, 2, 2])
    y_proba = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = compute_roc_auc(y_test, y_proba)

    assert result["roc_auc"][0] == 1.0
    assert result["roc_auc"][1] == 1.0
    assert result["roc_auc"][2] == 1.0
    assert result["macro_auc"] == 1.0


def test_compute_roc_auc_handles_degenerate_constant_probabilities():
    y_test = pd.Series([0, 0, 1, 1, 2, 2])
    # Classes 0 and 1 have varying, informative probabilities; class 2's
    # probability is constant zero for every sample (model never predicts it) —
    # roc_curve can't be computed for a constant column, so it should be flagged
    # as NaN rather than raising.
    y_proba = np.array(
        [
            [0.9, 0.05, 0.0],
            [0.8, 0.10, 0.0],
            [0.3, 0.70, 0.0],
            [0.2, 0.60, 0.0],
            [0.1, 0.20, 0.0],
            [0.05, 0.15, 0.0],
        ]
    )

    result = compute_roc_auc(y_test, y_proba)

    assert not np.isnan(result["roc_auc"][0])
    assert not np.isnan(result["roc_auc"][1])
    assert np.isnan(result["roc_auc"][2])
    assert not np.isnan(result["macro_auc"])  # nanmean ignores the degenerate class
