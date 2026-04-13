#!/bin/bash
# Smoke tests against the running API
BASE_URL="${1:-http://localhost:8000}"
echo "==> Testing API at: $BASE_URL"

echo ""
echo "--- /health ---"
curl -sf "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "--- /predict (Low Risk - Success expected) ---"
curl -sf -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "female_age": 26, "male_age": 28, "bmi": 22.0,
    "menstrual_regularity": "Regular", "pcos": "No",
    "stress_level": "Low", "smoking": "No",
    "alcohol_intake": "None", "sperm_count_million_per_ml": 75.0,
    "motility_pct": 80.0, "trying_duration_months": 6,
    "treatment_type": "IVF"
  }' | python3 -m json.tool

echo ""
echo "--- /predict (High Risk - Failure expected) ---"
curl -sf -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "female_age": 42, "male_age": 45, "bmi": 31.0,
    "menstrual_regularity": "Irregular", "pcos": "Yes",
    "stress_level": "High", "smoking": "Yes",
    "alcohol_intake": "High", "sperm_count_million_per_ml": 12.0,
    "motility_pct": 22.0, "trying_duration_months": 30,
    "treatment_type": "None"
  }' | python3 -m json.tool

echo ""
echo "==> Smoke tests complete."
