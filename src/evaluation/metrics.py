"""
Evaluation — ROC, PR curve, confusion matrix, feature importance, threshold sensitivity.
All plots saved to artifacts/plots/.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _plot_dir(config: dict) -> Path:
    d = Path(config["paths"]["plots_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_roc_curve(pipeline: Pipeline, X_test, y_test, config: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve -- Fertility Outcome Classifier")
    path = _plot_dir(config) / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved -> {path}")


def plot_pr_curve(pipeline: Pipeline, X_test, y_test, config: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve -- Fertility Outcome Classifier")
    path = _plot_dir(config) / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved -> {path}")


def plot_confusion_matrix(pipeline: Pipeline, X_test, y_test, config: dict):
    threshold = config["serving"]["decision_threshold"]
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Failure", "Success"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    path = _plot_dir(config) / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {path}")


def plot_feature_importance(pipeline: Pipeline, config: dict, top_n: int = 20):
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = clf.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="#2E86AB")
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Features -- Fertility Outcome Classifier")
    path = _plot_dir(config) / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance saved -> {path}")


def plot_threshold_sensitivity(pipeline: Pipeline, X_test, y_test, config: dict):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fn = ((preds == 0) & (y_test == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        rows.append({"threshold": round(t, 2), "recall": recall, "precision": precision, "f1": f1})

    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res.threshold, res.recall, label="Recall", color="#E84855", lw=2)
    ax.plot(res.threshold, res.precision, label="Precision", color="#2E86AB", lw=2)
    ax.plot(res.threshold, res.f1, label="F1", color="#3BB273", lw=2, linestyle="--")
    ax.axvline(config["serving"]["decision_threshold"], color="grey",
               linestyle=":", label=f"Chosen threshold ({config['serving']['decision_threshold']})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity -- Fertility Outcome Classifier")
    ax.legend()
    plt.tight_layout()
    path = _plot_dir(config) / "threshold_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Threshold sensitivity saved -> {path}")
    return res


def run_all_plots(pipeline: Pipeline, X_test, y_test, config: dict):
    plot_roc_curve(pipeline, X_test, y_test, config)
    plot_pr_curve(pipeline, X_test, y_test, config)
    plot_confusion_matrix(pipeline, X_test, y_test, config)
    plot_feature_importance(pipeline, config)
    plot_threshold_sensitivity(pipeline, X_test, y_test, config)
