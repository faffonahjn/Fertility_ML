"""
FastAPI serving layer -- Fertility Outcome Classifier v1.
Features: female_age, male_age, bmi, menstrual_regularity, pcos, stress_level,
          smoking, alcohol_intake, sperm_count, motility, trying_duration, treatment_type
Target: Pregnancy_Outcome (Success / Failure)
"""
import logging
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"
_model = None
DECISION_THRESHOLD = 0.40  # tuned for Failure recall


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _init_model():
    global _model
    if _model is None:
        config = _load_config()
        model_path = Path(config["paths"]["model_dir"]) / config["training"]["model_filename"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_model()
    logger.info("API ready.")
    yield


app = FastAPI(
    title="Fertility Outcome Classifier",
    description="Predicts pregnancy success probability from couples' health, lifestyle, and treatment features.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class FertilityRecord(BaseModel):
    female_age: int = Field(..., ge=18, le=50)
    male_age: int = Field(..., ge=18, le=70)
    bmi: float = Field(..., ge=10.0, le=60.0)
    menstrual_regularity: str
    pcos: str
    stress_level: str
    smoking: str
    alcohol_intake: str
    sperm_count_million_per_ml: float = Field(..., ge=0.0)
    motility_pct: float = Field(..., ge=0.0, le=100.0)
    trying_duration_months: int = Field(..., ge=0)
    treatment_type: str

    @field_validator("menstrual_regularity")
    @classmethod
    def val_menstrual(cls, v):
        if v not in {"Regular", "Irregular"}:
            raise ValueError("menstrual_regularity must be 'Regular' or 'Irregular'")
        return v

    @field_validator("pcos")
    @classmethod
    def val_pcos(cls, v):
        if v not in {"Yes", "No"}:
            raise ValueError("pcos must be 'Yes' or 'No'")
        return v

    @field_validator("stress_level")
    @classmethod
    def val_stress(cls, v):
        if v not in {"Low", "Medium", "High"}:
            raise ValueError("stress_level must be 'Low', 'Medium', or 'High'")
        return v

    @field_validator("smoking")
    @classmethod
    def val_smoking(cls, v):
        if v not in {"Yes", "No"}:
            raise ValueError("smoking must be 'Yes' or 'No'")
        return v

    @field_validator("alcohol_intake")
    @classmethod
    def val_alcohol(cls, v):
        if v not in {"None", "Moderate", "High"}:
            raise ValueError("alcohol_intake must be 'None', 'Moderate', or 'High'")
        return v

    @field_validator("treatment_type")
    @classmethod
    def val_treatment(cls, v):
        if v not in {"None", "Medication", "IVF"}:
            raise ValueError("treatment_type must be 'None', 'Medication', or 'IVF'")
        return v


class PredictionResponse(BaseModel):
    pregnancy_success: int
    success_probability: float
    outcome_label: str
    risk_level: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_records: int
    success_count: int
    failure_count: int


def _record_to_df(record: FertilityRecord) -> pd.DataFrame:
    """Convert API record to DataFrame with correct column names for pipeline."""
    d = record.model_dump()
    return pd.DataFrame([{
        "Female_Age": d["female_age"],
        "Male_Age": d["male_age"],
        "BMI": d["bmi"],
        "Menstrual_Regularity": d["menstrual_regularity"],
        "PCOS": d["pcos"],
        "Stress_Level": d["stress_level"],
        "Smoking": d["smoking"],
        "Alcohol_Intake": d["alcohol_intake"],
        "Sperm_Count_Million_per_ml": d["sperm_count_million_per_ml"],
        "Motility_%": d["motility_pct"],
        "Trying_Duration_Months": d["trying_duration_months"],
        "Treatment_Type": d["treatment_type"],
    }])


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "Low Risk"
    elif prob >= 0.50:
        return "Moderate Risk"
    else:
        return "High Risk"


def _predict_single(record: FertilityRecord) -> PredictionResponse:
    _init_model()
    t0 = time.perf_counter()
    from src.features.engineer import add_engineered_features
    df = _record_to_df(record)
    df = add_engineered_features(df)
    prob = float(_model.predict_proba(df)[0][1])
    label = int(prob >= DECISION_THRESHOLD)
    return PredictionResponse(
        pregnancy_success=label,
        success_probability=round(prob, 4),
        outcome_label="Success" if label else "Failure",
        risk_level=_risk_level(prob),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "version": "1.0.0",
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(record: FertilityRecord):
    return _predict_single(record)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(records: List[FertilityRecord]):
    if len(records) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit: 500 records")
    predictions = [_predict_single(r) for r in records]
    success_count = sum(p.pregnancy_success for p in predictions)
    return BatchPredictionResponse(
        predictions=predictions,
        total_records=len(predictions),
        success_count=success_count,
        failure_count=len(predictions) - success_count,
    )
