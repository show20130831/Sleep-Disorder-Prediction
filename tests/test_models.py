from sleep_disorder.models import MODEL_KEYS, build_preprocessor, get_model_registry
from sleep_disorder.preprocessing import clean_dataset, split_features_and_target


def test_preprocessor_fits_and_transforms(raw_df):
    cleaned = clean_dataset(raw_df)
    X, _ = split_features_and_target(cleaned)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)


def test_get_model_registry_has_three_models():
    registry = get_model_registry()
    assert set(registry) == set(MODEL_KEYS)
    for spec in registry.values():
        assert spec.estimator_step in spec.pipeline.named_steps
        assert "preprocess" in spec.pipeline.named_steps
