import pandas as pd

from sleep_disorder.descriptive_analysis import compute_categorical_summary, compute_numeric_summary


def test_compute_numeric_summary_includes_anova_pvalue():
    df = pd.DataFrame(
        {
            "Age": [20, 22, 21, 40, 42, 41, 60, 62, 61],
            "SleepDisorder_3class": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )
    result = compute_numeric_summary(df, numeric_cols=["Age"], target_col="SleepDisorder_3class")
    assert result.loc[0, "Variable"] == "Age"
    assert result.loc[0, "ANOVA_p_value"] < 0.05


def test_compute_categorical_summary_flags_sparse_assumption_violation():
    df = pd.DataFrame(
        {
            "Occupation_Manager": [1, 0, 0, 0, 0, 0],
            "SleepDisorder_3class": [0, 0, 1, 1, 2, 2],
        }
    )
    result = compute_categorical_summary(
        df, categorical_dummy_cols=["Occupation_Manager"], target_col="SleepDisorder_3class"
    )
    assert bool(result.loc[0, "Assumption_violated (expected count < 5)"]) is True
