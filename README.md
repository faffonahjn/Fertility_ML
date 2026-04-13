# 🧬 Fertility Outcome Classifier

> **End-to-end production ML pipeline** predicts pregnancy success probability from couples' health, lifestyle, and fertility treatment features, with a 4-tab clinical Streamlit dashboard and FastAPI REST API.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-24%20passing-brightgreen)](tests/)
[![AUC](https://img.shields.io/badge/Test%20AUC-0.950-blue)](artifacts/metrics/)

---

## 📌 Project Overview

This project demonstrates a **production-grade clinical ML system** for reproductive health analytics. It predicts whether a couple's fertility journey will result in pregnancy success, based on comprehensive demographic, health, lifestyle, and treatment features from both partners.

**Clinical Problem:** Infertility affects an estimated 1 in 6 couples worldwide. Early identification of couples at high risk of treatment failure enables timely intervention IVF escalation, lifestyle modification counseling, or specialist referral — before the treatment window closes. Female age is the single most critical biological factor, yet clinical decisions rarely integrate male-side sperm metrics with female-side indicators in a unified predictive framework.

**What makes this project different:**

- **Informative NaN strategy** — 62.5% of `Treatment_Type` values and 32.4% of `Alcohol_Intake` values were null. Rather than statistical imputation, these were domain-filled as `"None"` clinically they mean *no treatment received* and *no alcohol consumed*. This is a production data engineering decision, not a convenience shortcut.
- **Engineered clinical interaction** — `Female_Age × Motility%` captures age-adjusted reproductive potential. As female age increases, sperm motility's compensatory effect compounds, a relationship neither feature captures independently.
- **Clinically tuned threshold** — Decision threshold set to **0.40** (not the default 0.50) to maximize recall for the Failure class. Missing a high-risk couple delays intervention and closes the treatment window.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| CV AUC (5-fold) | 0.946 +/- 0.012 |
| **Test AUC** | **0.9504** |
| Average Precision | 0.9802 |
| Decision Threshold | **0.40** (Failure recall priority) |
| Tests Passing | **24 / 24** |

---

## 🏗️ System Architecture

```
+---------------------------------------------------------+
|                      DATA LAYER                         |
|  data/raw/ -> loader.py (informative NaN fills)         |
|           -> engineer.py (Female_Age x Motility)        |
+------------------------+--------------------------------+
                         |
+------------------------v--------------------------------+
|                    TRAINING LAYER                       |
|  train_pipeline.py -> XGBoost Pipeline                 |
|  -> fertility_classifier_v1.pkl                         |
|  Evaluation: ROC | PR | SHAP | Threshold Sensitivity    |
+------------------------+--------------------------------+
                         |
+------------------------v--------------------------------+
|                    SERVING LAYER                        |
|  FastAPI -> POST /predict | POST /predict/batch         |
|  Streamlit -> Single | Batch | EDA | Model Info         |
+------------------------+--------------------------------+
                         |
+------------------------v--------------------------------+
|                 INFRASTRUCTURE LAYER                    |
|  Docker (multi-stage) -> Compose -> Azure Container Apps|
+---------------------------------------------------------+
```

---

## 🔬 Key Engineering Decisions

### 1. Informative NaN Strategy

Most data science tutorials fill NaN values with the column mean, median, or mode. For this clinical dataset, that would be **clinically wrong**.

| Column | NaN Count | Fill Value | Clinical Meaning |
|---|---|---|---|
| `Treatment_Type` | 500 (62.5%) | `"None"` | Couple received no fertility treatment |
| `Alcohol_Intake` | 259 (32.4%) | `"None"` | No alcohol consumption reported |

Imputing `Treatment_Type` with `"Medication"` would falsely claim 500 couples received treatment they never had. The NaN is the data — it carries meaning.

### 2. Engineered Interaction Feature

```python
df["Female_Age_x_Motility"] = df["Female_Age"] * df["Motility_%"]
```

As a woman's ovarian reserve declines with age, the quality of sperm (motility) plays a relatively larger compensatory role. A 25-year-old woman with 30% motility has a fundamentally different prognosis than a 42-year-old with the same motility. This interaction is clinically motivated by reproductive endocrinology literature.

### 3. Clinical Threshold Rationale

Default threshold = 0.50 (maximizes F1). Clinical threshold = **0.40** (maximizes Failure recall).

In fertility counseling, a **False Negative** (predicting Success for a couple who will fail) delays intervention and loses the treatment window — especially critical for women over 38 where each month matters. A **False Positive** (flagging a couple who would succeed) causes unnecessary anxiety but no physical harm. The asymmetric cost justifies a lower threshold.

### 4. Class Imbalance via scale_pos_weight

```yaml
scale_pos_weight: 0.37   # 218 Failure / 582 Success = 0.37
```

Rather than SMOTE (which generates synthetic patients — clinically meaningless) or undersampling (which discards real data), XGBoost's built-in `scale_pos_weight` mathematically upweights the minority Failure class during training. Same effect, no artificial data.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+ · Docker Desktop · Anaconda / pip

### 1. Clone and Setup

```bash
git clone https://github.com/faffonahjn/fertility-outcome-classifier.git
cd fertility-outcome-classifier

conda activate ml_env
pip install -r requirements.txt
```

### 2. Train

```bash
python pipelines/train_pipeline.py
```

Expected output:
```
FERTILITY ML - TRAINING PIPELINE START
Filled 500 NaNs in 'Treatment_Type' with 'None' (informative missing)
Filled 259 NaNs in 'Alcohol_Intake' with 'None' (informative missing)
Engineered feature added: Female_Age_x_Motility
CV AUC: 0.9463 +/- 0.0121
Test AUC: 0.9504 | AP: 0.9802
TRAINING COMPLETE
```

### 3. Batch Predict

```bash
python pipelines/predict_pipeline.py \
  --input data/raw/Fertility_Health_Dataset_2026.csv \
  --output data/processed/predictions.csv
```

### 4. Serve Locally

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
# API docs -> http://localhost:8000/docs
```

### 5. Full Stack (Docker)

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

### 6. Run Tests

```bash
pytest tests/ -v
# 24 passed
```

---

## 🌐 API Reference

**Base URL:** `http://localhost:8000`

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "decision_threshold": 0.40
}
```

### POST /predict

**Request:**
```json
{
  "female_age": 30,
  "male_age": 32,
  "bmi": 24.5,
  "menstrual_regularity": "Regular",
  "pcos": "No",
  "stress_level": "Low",
  "smoking": "No",
  "alcohol_intake": "None",
  "sperm_count_million_per_ml": 65.0,
  "motility_pct": 72.0,
  "trying_duration_months": 10,
  "treatment_type": "Medication"
}
```

**Response:**
```json
{
  "pregnancy_success": 1,
  "success_probability": 0.8741,
  "outcome_label": "Success",
  "risk_level": "Low Risk",
  "latency_ms": 3.12
}
```

### POST /predict/batch

Accepts up to 500 couple records. Returns individual predictions plus aggregate counts.

```json
{
  "predictions": [...],
  "total_records": 50,
  "success_count": 38,
  "failure_count": 12
}
```

### Field Validation

| Field | Type | Valid Values |
|---|---|---|
| `female_age` | int | 18 - 50 |
| `male_age` | int | 18 - 70 |
| `bmi` | float | 10.0 - 60.0 |
| `menstrual_regularity` | str | `Regular`, `Irregular` |
| `pcos` | str | `Yes`, `No` |
| `stress_level` | str | `Low`, `Medium`, `High` |
| `smoking` | str | `Yes`, `No` |
| `alcohol_intake` | str | `None`, `Moderate`, `High` |
| `sperm_count_million_per_ml` | float | >= 0 |
| `motility_pct` | float | 0 - 100 |
| `trying_duration_months` | int | >= 0 |
| `treatment_type` | str | `None`, `Medication`, `IVF` |

---

## 📁 Project Structure

```
ML_Pregnancy_Prediction/
+-- artifacts/
|   +-- metrics/           evaluation_metrics.json
|   +-- models/            fertility_classifier_v1.pkl
|   +-- plots/             ROC | PR | CM | Feature Importance | Threshold Sensitivity
+-- configs/
|   +-- config.yaml        Single source of truth for all parameters
+-- data/
|   +-- raw/               Fertility_Health_Dataset_2026.csv
|   +-- processed/         predictions.csv (batch inference output)
+-- docker/
|   +-- Dockerfile         Multi-stage API image (Python 3.11-slim)
|   +-- Dockerfile.streamlit
|   +-- docker-compose.yml
+-- docs/
|   +-- architecture.md
+-- notebooks/
|   +-- exploratory/       01_eda.ipynb
|   +-- modeling/          02_modeling.ipynb
|   +-- evaluation/        03_evaluation.ipynb
+-- pipelines/
|   +-- train_pipeline.py
|   +-- predict_pipeline.py
+-- scripts/
|   +-- setup_azure.sh     Full Azure Container Apps provisioning
|   +-- retrain.sh         AUC-gated retrain + Docker rebuild
|   +-- test_api.sh        API smoke tests
+-- src/
|   +-- data/              loader.py (NaN fills, schema validation)
|   +-- features/          engineer.py (interaction feature + preprocessor)
|   +-- models/            trainer.py (XGBoost Pipeline + CV + joblib)
|   +-- evaluation/        metrics.py (5 evaluation plots)
|   +-- serving/           api.py (FastAPI)
|   +-- utils/             validators.py + logger.py
+-- streamlit_app/
|   +-- app.py             4-tab clinical dashboard
+-- tests/
|   +-- unit/              test_features | test_trainer | test_validators
|   +-- integration/       test_api
+-- Makefile
+-- requirements.txt
+-- README.md
```

---

## 🖥️ Streamlit Dashboard

The clinical dashboard provides four tabs:

| Tab | Description |
|---|---|
| 🔍 Single Prediction | Couple form with live success probability, risk level, and probability gauge |
| 📋 Batch Prediction | CSV upload, bulk scoring, downloadable predictions with risk tiers |
| 📊 EDA Dashboard | Age distributions, success rates by feature, motility scatter, correlation heatmap |
| ℹ️ Model Info | Architecture, metrics, NaN audit table, threshold rationale |

The sidebar shows live API health status, current threshold, and model version.

---

## 🧪 Test Suite (24 Tests)

```
tests/unit/test_features.py     (7 tests)
  - Engineered feature created and mathematically correct
  - Preprocessor builds without error
  - Feature/target split removes ID and target columns
  - Informative NaN fill verified

tests/unit/test_validators.py   (8 tests)
  - Valid record passes
  - Each invalid field type caught and reported

tests/unit/test_trainer.py      (2 tests)
  - Pipeline builds with correct named steps
  - Train returns valid metrics dict

tests/integration/test_api.py   (7 tests)
  - Health endpoint returns 200
  - Low-risk couple predicts Success
  - High-risk couple predicts Failure
  - Batch endpoint with correct aggregate counts
  - Invalid field rejected with 422
  - Batch limit enforced at 501 records
```

---

## ☁️ Azure Deployment

```bash
az login
bash scripts/setup_azure.sh
```

Provisions Azure Resource Group, Container Registry, Container Apps environment, and deploys the API with external HTTPS ingress and auto-scaling (1-3 replicas).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost 2.0 · scikit-learn 1.4 |
| Pipeline | sklearn Pipeline · ColumnTransformer |
| API | FastAPI · Uvicorn · Pydantic v2 |
| Dashboard | Streamlit · Matplotlib · Seaborn |
| Serialization | Joblib |
| Containerization | Docker multi-stage · Compose |
| Cloud | Azure Container Apps · Azure Container Registry |
| Testing | pytest · FastAPI TestClient |
| Config | YAML-driven |

---

## 📋 Dataset

**Fertility Health Dataset 2026** — 800 synthetic couple records, 14 features, binary target.

| Feature | Type | Clinical Meaning |
|---|---|---|
| `Female_Age` | Numeric | Primary biological predictor -- fertility declines with age |
| `Male_Age` | Numeric | Secondary predictor -- sperm quality declines with age |
| `BMI` | Numeric | High/low BMI affects hormonal balance |
| `Menstrual_Regularity` | Categorical | Proxy for ovulation regularity |
| `PCOS` | Categorical | Major hormonal fertility condition |
| `Stress_Level` | Categorical | Stress affects reproductive hormones |
| `Smoking` | Categorical | Reduces ovarian reserve and sperm quality |
| `Alcohol_Intake` | Categorical + NaN | NaN = no consumption (informative) |
| `Sperm_Count_Million_per_ml` | Numeric | WHO threshold: > 15M/ml normal |
| `Motility_%` | Numeric | WHO threshold: > 40% normal |
| `Trying_Duration_Months` | Numeric | Duration of active conception attempts |
| `Treatment_Type` | Categorical + NaN | NaN = no treatment received (informative) |
| `Pregnancy_Outcome` | **Target** | Success (1) / Failure (0) |

---

## 👤 Author

**Francis Affonah**
Clinical Data Scientist · ML Engineer · Registered Nurse

[![GitHub](https://img.shields.io/badge/GitHub-faffonahjn-black?logo=github)](https://github.com/faffonahjn)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Francis_Affonah-blue?logo=linkedin)](https://linkedin.com/in/francis-affonah-23745a205/)
[![Live API](https://img.shields.io/badge/Live%20API-Insurance%20Risk%20Classifier-success?logo=microsoftazure)](https://insurance-risk-api.redsand-37d94e81.eastus.azurecontainerapps.io/docs)

> *"So built we the wall... for the people had a mind to work."* -- Nehemiah 4:6

---

## 📄 License

MIT License -- see [LICENSE](LICENSE) for details.
