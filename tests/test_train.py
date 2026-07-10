from sleep_disorder import config
from sleep_disorder.data_loader import load_raw_data
from sleep_disorder.preprocessing import clean_dataset, split_features_and_target
from sleep_disorder.train import split_train_test


def test_split_train_test_is_stratified_and_sized_correctly():
    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    X, y = split_features_and_target(cleaned)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - config.TEST_SIZE) < 0.02

    train_proportions = y_train.value_counts(normalize=True).sort_index()
    test_proportions = y_test.value_counts(normalize=True).sort_index()
    assert (train_proportions - test_proportions).abs().max() < 0.05


def test_split_train_test_is_reproducible():
    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    X, y = split_features_and_target(cleaned)

    X_train_1, X_test_1, _, _ = split_train_test(X, y)
    X_train_2, X_test_2, _, _ = split_train_test(X, y)

    assert X_train_1.index.equals(X_train_2.index)
    assert X_test_1.index.equals(X_test_2.index)
