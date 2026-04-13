"""Unit tests -- feature engineering."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_config, load_raw_data, preprocess_raw, split_features_target
from src.features.engineer import build_preprocessor, add_engineered_features


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


@pytest.fixture
def sample_data(config):
    df = load_raw_data(config["paths"]["raw_data"])
    df = preprocess_raw(df, config)
    return df.head(50)


def test_engineered_feature_created(sample_data):
    df = add_engineered_features(sample_data)
    assert "Female_Age_x_Motility" in df.columns


def test_engineered_feature_values(sample_data):
    df = add_engineered_features(sample_data)
    expected = df["Female_Age"] * df["Motility_%"]
    pd.testing.assert_series_equal(df["Female_Age_x_Motility"], expected, check_names=False)


def test_preprocessor_builds(config):
    assert build_preprocessor(config) is not None


def test_feature_target_split(config, sample_data):
    X, y = split_features_target(sample_data, config)
    assert "Pregnancy_Outcome" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_no_id_column(config, sample_data):
    X, y = split_features_target(sample_data, config)
    assert "Couple_ID" not in X.columns


def test_informative_nan_filled(config):
    df = load_raw_data(config["paths"]["raw_data"])
    df = preprocess_raw(df, config)
    assert df["Treatment_Type"].isna().sum() == 0
    assert df["Alcohol_Intake"].isna().sum() == 0
    assert "None" in df["Treatment_Type"].values
    assert "None" in df["Alcohol_Intake"].values


def test_preprocessor_transform_shape(config, sample_data):
    X, y = split_features_target(sample_data, config)
    X = add_engineered_features(X)
    preprocessor = build_preprocessor(config)
    X_t = preprocessor.fit_transform(X)
    assert X_t.shape[0] == len(X)
    assert X_t.shape[1] > 0
