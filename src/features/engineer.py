"""
Feature engineering — adds clinical interaction feature, builds sklearn preprocessor.
Key engineered feature: Female_Age_x_Motility — captures age-adjusted reproductive potential.
"""
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-motivated interaction features.
    Female_Age_x_Motility: as female age increases, sperm motility's protective
    effect compounds — this interaction captures age-adjusted reproductive potential.
    """
    df = df.copy()
    df["Female_Age_x_Motility"] = df["Female_Age"] * df["Motility_%"]
    logger.info("Engineered feature added: Female_Age_x_Motility")
    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    cat_features = config["data"]["categorical_features"]
    num_features = config["data"]["numeric_features"]
    bin_features = config["data"].get("binary_features", [])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    transformers = [
        ("cat", categorical_pipe, cat_features),
        ("num", numeric_pipe, num_features),
    ]
    if bin_features:
        transformers.append(("bin", "passthrough", bin_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        f"Preprocessor built | cat={len(cat_features)} | "
        f"num={len(num_features)} | bin={len(bin_features)}"
    )
    return preprocessor
