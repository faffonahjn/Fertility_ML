"""
predict_pipeline.py -- batch inference on a CSV file.

Usage:
    python pipelines/predict_pipeline.py --input data/raw/Fertility_Health_Dataset_2026.csv
"""
import argparse
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import load_config, preprocess_raw
from src.features.engineer import add_engineered_features
from src.models.trainer import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISION_THRESHOLD = 0.40


def main(input_path: str, output_path: str, config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    model_path = Path(config["paths"]["model_dir"]) / config["training"]["model_filename"]

    logger.info(f"Loading model from {model_path}")
    pipeline = load_pipeline(str(model_path))

    logger.info(f"Loading input data from {input_path}")
    df = pd.read_csv(input_path)
    df = preprocess_raw(df, config)

    drop_cols = config["data"]["drop_columns"] + [config["data"]["target"]]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = add_engineered_features(X)

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)

    df["predicted_success_prob"] = probs.round(4)
    df["predicted_outcome"] = preds
    df["outcome_label"] = df["predicted_outcome"].map({1: "Success", 0: "Failure"})
    df["risk_level"] = pd.cut(
        probs,
        bins=[0, 0.5, 0.75, 1.0],
        labels=["High Risk", "Moderate Risk", "Low Risk"]
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved -> {output_path} | Success: {preds.sum()}/{len(preds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/predictions.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.input, args.output, args.config)
