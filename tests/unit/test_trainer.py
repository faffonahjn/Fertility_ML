"""Unit tests -- model trainer."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.loader import load_config, load_raw_data, preprocess_raw, split_features_target
from src.models.trainer import build_pipeline, train
from sklearn.model_selection import train_test_split


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


def test_pipeline_builds(config):
    pipeline = build_pipeline(config)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_returns_valid_metrics(config):
    df = load_raw_data(config["paths"]["raw_data"])
    df = preprocess_raw(df, config)
    X, y = split_features_target(df, config)
    _, metrics, X_test, y_test = train(X, y, config)
    assert "test_roc_auc" in metrics
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
