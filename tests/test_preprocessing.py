import pytest

from sleep_disorder.preprocessing import (
    clean_dataset,
    encode_target,
    merge_bmi_categories,
    split_blood_pressure,
    split_features_and_target,
)


def test_split_blood_pressure(raw_df):
    result = split_blood_pressure(raw_df)
    assert "Blood Pressure" not in result.columns
    assert list(result["Systolic_BP"]) == [126, 125, 140, 118]
    assert list(result["Diastolic_BP"]) == [83, 80, 95, 76]


def test_encode_target_maps_known_labels(raw_df):
    result = encode_target(raw_df)
    assert list(result["SleepDisorder_3class"]) == [0, 0, 2, 1]


def test_encode_target_raises_on_unknown_label(raw_df):
    raw_df.loc[0, "Sleep Disorder"] = "Narcolepsy"
    with pytest.raises(ValueError):
        encode_target(raw_df)


def test_merge_bmi_categories_collapses_normal_weight(raw_df):
    result = merge_bmi_categories(raw_df)
    assert "Normal Weight" not in result["BMI Category"].values
    assert result["BMI Category"].value_counts()["Normal"] == 2


def test_clean_dataset_end_to_end(raw_df):
    cleaned = clean_dataset(raw_df)
    assert {"Systolic_BP", "Diastolic_BP", "SleepDisorder_3class"} <= set(cleaned.columns)
    assert "Blood Pressure" not in cleaned.columns
    assert "Normal Weight" not in cleaned["BMI Category"].values


def test_split_features_and_target_drops_id_and_raw_labels(raw_df):
    cleaned = clean_dataset(raw_df)
    X, y = split_features_and_target(cleaned)
    assert "Person ID" not in X.columns
    assert "Sleep Disorder" not in X.columns
    assert "SleepDisorder_3class" not in X.columns
    assert list(y) == [0, 0, 2, 1]
