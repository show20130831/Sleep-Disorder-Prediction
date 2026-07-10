from sleep_disorder import config


def test_raw_data_path_exists():
    assert config.RAW_DATA_PATH.exists()


def test_class_names_match_label_map():
    assert len(config.CLASS_NAMES) == len(config.SLEEP_DISORDER_LABEL_MAP)
    for name, index in config.SLEEP_DISORDER_LABEL_MAP.items():
        assert config.CLASS_NAMES[index] == name


def test_bmi_merge_targets_are_not_self_referential():
    for source, target in config.BMI_CATEGORY_MERGE_MAP.items():
        assert source != target


def test_test_size_is_a_valid_fraction():
    assert 0.0 < config.TEST_SIZE < 1.0


def test_pipeline_config_hyperparameters_share_the_global_random_state():
    assert config.PIPELINE_CONFIG.random_forest.random_state == config.RANDOM_STATE
    assert config.PIPELINE_CONFIG.xgboost.random_state == config.RANDOM_STATE
    assert config.PIPELINE_CONFIG.logistic_regression.random_state == config.RANDOM_STATE
    assert config.PIPELINE_CONFIG.smote.random_state == config.RANDOM_STATE
