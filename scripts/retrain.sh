#!/bin/bash
# AUC-gated retrain + Docker rebuild
set -euo pipefail

MIN_AUC=0.85

echo "==> Retraining..."
python pipelines/train_pipeline.py --config configs/config.yaml

AUC=$(python3 -c "
import json
with open('artifacts/metrics/evaluation_metrics.json') as f:
    m = json.load(f)
print(m['test_roc_auc'])
")

echo "==> Test AUC: $AUC | Minimum: $MIN_AUC"
PASS=$(python3 -c "print('yes' if float('$AUC') >= float('$MIN_AUC') else 'no')")

if [ "$PASS" = "yes" ]; then
    echo "==> AUC gate PASSED. Rebuilding Docker image..."
    docker build -f docker/Dockerfile -t fertility-outcome-api:latest .
    echo "==> Done. Run 'make azure-deploy' to push."
else
    echo "==> AUC gate FAILED ($AUC < $MIN_AUC). Aborting."
    exit 1
fi
