#!/usr/bin/env bash
# Deploy Medicare Classifier FastAPI to Azure Container Apps (API only).
# Prerequisites: az login, GHCR image public (or set GHCR_USER + GHCR_TOKEN).

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-claims-intelligence}"
LOCATION="${LOCATION:-eastus}"
ENV_NAME="${ENV_NAME:-claims-env}"
APP_NAME="${APP_NAME:-medicare-classifier-api}"
IMAGE="${IMAGE:-ghcr.io/deepakguptaaiml/medicare-classifier:latest}"

echo "==> Resource group: $RESOURCE_GROUP ($LOCATION)"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "==> Container Apps environment: $ENV_NAME"
az containerapp env create \
  --name "$ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none 2>/dev/null || echo "    (environment may already exist)"

REGISTRY_ARGS=()
if [[ -n "${GHCR_TOKEN:-}" ]]; then
  REGISTRY_ARGS=(
    --registry-server ghcr.io
    --registry-username "${GHCR_USER:-DeepakGuptaaiml}"
    --registry-password "$GHCR_TOKEN"
  )
  echo "==> Using private GHCR credentials"
else
  echo "==> Assuming public GHCR image (no registry credentials)"
fi

echo "==> Container app: $APP_NAME"
if az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  echo "    Updating existing app..."
  az containerapp update \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$IMAGE" \
    --output none
else
  az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "$IMAGE" \
    --target-port 8000 \
    --ingress external \
    --cpu 0.5 \
    --memory 1.0Gi \
    --min-replicas 1 \
    --max-replicas 2 \
    "${REGISTRY_ARGS[@]}" \
    --output none
fi

FQDN=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv)

echo ""
echo "============================================"
echo " Medicare Classifier API deployed"
echo "============================================"
echo " URL:      https://${FQDN}"
echo " Health:   https://${FQDN}/health"
echo " Swagger:  https://${FQDN}/docs"
echo ""
echo "Test:"
echo "  curl https://${FQDN}/health"
echo "============================================"
