"""
Data loader — reads raw CSV, validates schema, fills informative NaNs,
encodes target, returns clean DataFrame.
"""
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Female_Age", "Male_Age", "BMI", "Menstrual_Regularity", "PCOS",
    "Stress_Level", "Smoking", "Alcohol_Intake", "Sperm_Count_Million_per_ml",
    "Motility_%", "Trying_Duration_Months", "Treatment_Type", "Pregnancy_Outcome",
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} records from {path}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def preprocess_raw(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Fill informative NaNs with domain-meaningful values before feature engineering.
    - Treatment_Type NaN = couple received no treatment
    - Alcohol_Intake NaN = no alcohol consumption reported
    """
    nan_fills = config["data"]["informative_nan_fills"]
    for col, fill_val in nan_fills.items():
        n_filled = df[col].isna().sum()
        if n_filled > 0:
            df[col] = df[col].fillna(fill_val)
            logger.info(f"Filled {n_filled} NaNs in '{col}' with '{fill_val}' (informative missing)")

    return df


def split_features_target(df: pd.DataFrame, config: dict):
    target_col = config["data"]["target"]
    positive_class = config["data"]["positive_class"]
    drop_cols = [c for c in config["data"]["drop_columns"] if c in df.columns]

    # Encode target: Success=1, Failure=0
    y = (df[target_col] == positive_class).astype(int)
    X = df.drop(columns=[target_col] + drop_cols)

    pos_rate = y.mean()
    logger.info(
        f"Features: {X.shape[1]} | Target: {target_col} | "
        f"Positive ({positive_class}): {pos_rate:.2%}"
    )
    return X, y
