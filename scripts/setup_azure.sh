#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_azure.sh — provisions Azure infrastructure for Fertility Outcome API
# Usage: bash scripts/setup_azure.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RESOURCE_GROUP="rg-fertility-ml"
LOCATION="eastus"
ACR_NAME="fertilitymlacr"
APP_ENV="fertility-ml-env"
APP_NAME="fertility-outcome-api"
IMAGE="$ACR_NAME.azurecr.io/fertility-outcome-api:latest"

echo "==> Creating Resource Group: $RESOURCE_GROUP"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" --sku Basic --admin-enabled true

echo "==> Building and pushing Docker image"
az acr login --name "$ACR_NAME"
docker build -f docker/Dockerfile -t fertility-outcome-api:latest .
docker tag fertility-outcome-api:latest "$IMAGE"
docker push "$IMAGE"

echo "==> Creating Container Apps Environment"
az containerapp env create \
  --name "$APP_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "==> Deploying Container App"
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$APP_ENV" \
  --image "$IMAGE" \
  --registry-server "$ACR_NAME.azurecr.io" \
  --registry-username "$(az acr credential show -n $ACR_NAME --query username -o tsv)" \
  --registry-password "$(az acr credential show -n $ACR_NAME --query passwords[0].value -o tsv)" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi

echo ""
echo "==> Deployment complete."
APP_URL=$(az containerapp show --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)
echo "API URL: https://$APP_URL"
echo "Docs:    https://$APP_URL/docs"
echo "Health:  https://$APP_URL/health"
