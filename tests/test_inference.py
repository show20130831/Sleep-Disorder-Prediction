from sleep_disorder.data_loader import load_raw_data
from sleep_disorder.inference import predict_one, prepare_record
from sleep_disorder.models import get_model_registry
from sleep_disorder.preprocessing import clean_dataset, split_features_and_target


def test_prepare_record_splits_bp_and_merges_bmi():
    record = {
        "Gender": "Male",
        "Age": 30,
        "Occupation": "Nurse",
        "BMI Category": "Normal Weight",
        "Blood Pressure": "126/83",
    }
    df = prepare_record(record)
    assert df.loc[0, "Systolic_BP"] == 126
    assert df.loc[0, "Diastolic_BP"] == 83
    assert df.loc[0, "BMI Category"] == "Normal"
    assert "Blood Pressure" not in df.columns


def test_predict_one_returns_class_and_probabilities():
    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    X, y = split_features_and_target(cleaned)

    pipeline = get_model_registry()["random_forest"].pipeline
    pipeline.fit(X, y)

    record = {
        "Gender": "Male",
        "Age": 43,
        "Occupation": "Nurse",
        "Sleep Duration": 6.0,
        "Quality of Sleep": 6,
        "Physical Activity Level": 45,
        "Stress Level": 7,
        "BMI Category": "Overweight",
        "Blood Pressure": "135/88",
        "Heart Rate": 78,
        "Daily Steps": 5200,
    }
    result = predict_one(pipeline, record)

    assert result["prediction"] in {"None", "Insomnia", "Sleep Apnea"}
    assert set(result["probabilities"]) == {"None", "Insomnia", "Sleep Apnea"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
