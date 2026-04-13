# Fertility Outcome Classifier

> **End-to-end clinical ML pipeline** -- predicts pregnancy success probability from couples' health, lifestyle, and fertility treatment features.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-24%20passing-brightgreen)](tests/)
[![AUC](https://img.shields.io/badge/Test%20AUC-0.950-blue)](artifacts/metrics/)

---

## Project Overview

This project demonstrates a **production-grade clinical ML system** for reproductive health analytics. It predicts whether a couple's fertility journey will result in pregnancy success, based on demographic, health, lifestyle, and treatment features.

**Clinical Problem:** Infertility affects millions of couples worldwide. Early identification of couples at high risk of treatment failure enables timely intervention — IVF escalation, lifestyle modification, or specialist referral.

**Key engineering decisions:**
- Informative NaN strategy: `Treatment_Type` and `Alcohol_Intake` NaN values are domain-filled rather than statistically imputed
- Engineered feature: `Female_Age x Motility` — captures age-adjusted reproductive potential
- Clinical threshold: **0.40** (not default 0.50) — maximizes recall for Failure class

---

## Model Performance

| Metric | Value |
|---|---|
| CV AUC (5-fold) | 0.946 +/- 0.012 |
| **Test AUC** | **0.950** |
| Average Precision | 0.980 |
| Decision Threshold | **0.40** (Failure recall) |

---

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/faffonahjn/fertility-outcome-classifier.git
cd fertility-outcome-classifier
conda activate ml_env
pip install -r requirements.txt

# 2. Train
python pipelines/train_pipeline.py

# 3. Serve locally
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# 4. Full stack (Docker)
docker compose -f docker/docker-compose.yml up --build

# 5. Tests
pytest tests/ -v
```

---

## API Reference

**Base URL:** `http://localhost:8000`

### POST /predict

```json
{
  "female_age": 30, "male_age": 32, "bmi": 24.5,
  "menstrual_regularity": "Regular", "pcos": "No",
  "stress_level": "Low", "smoking": "No",
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

---

## Project Structure

```
FERTILITY_ML/
+-- artifacts/          models, metrics, plots
+-- configs/            config.yaml
+-- data/raw/           Fertility_Health_Dataset_2026.csv
+-- docker/             Dockerfile, Dockerfile.streamlit, docker-compose.yml
+-- docs/               architecture.md
+-- notebooks/          01_eda, 02_modeling, 03_evaluation
+-- pipelines/          train_pipeline.py, predict_pipeline.py
+-- scripts/            setup_azure.sh, retrain.sh, test_api.sh
+-- src/
|   +-- data/           loader.py (informative NaN fills)
|   +-- features/       engineer.py (Female_Age x Motility)
|   +-- models/         trainer.py
|   +-- evaluation/     metrics.py (5 plots)
|   +-- serving/        api.py (FastAPI)
|   +-- utils/          validators.py, logger.py
+-- streamlit_app/      app.py (4-tab clinical dashboard)
+-- tests/              24 tests (unit + integration)
+-- Makefile
+-- requirements.txt
```

---

## Author

**Francis Affonah** -- Data Scientist | Healthcare Analytics | ML Engineering

[![GitHub](https://img.shields.io/badge/GitHub-faffonahjn-black?logo=github)](https://github.com/faffonahjn)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Francis_Affonah-blue?logo=linkedin)](https://linkedin.com/in/francis-affonah-23745a205/)
