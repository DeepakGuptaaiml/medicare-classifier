#!/usr/bin/env bash
# Bootstrap Azure Blob containers + AML datastores for ML training input data.
#
# Express Script (org pipeline) writes parquet to these containers.
# AML jobs on claims-compute read via registered datastores.
#
# Prerequisites:
#   az login
#   pip install -r requirements-register.txt   (for register_training_datastore.py)
#
# Usage:
#   chmod +x scripts/setup-training-datastore.sh
#   ./scripts/setup-training-datastore.sh
#
# Optional env:
#   RESOURCE_GROUP=rg-claims-intelligence
#   STORAGE_ACCOUNT=claimsmlstorage4cd105406   # existing AML workspace storage
#   WORKSPACE_NAME=claims-ml-workspace

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-claims-intelligence}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-${TRAINING_STORAGE_ACCOUNT:-claimsmlstorage4cd105406}}"
WORKSPACE_NAME="${WORKSPACE_NAME:-claims-ml-workspace}"
MEDICARE_CONTAINER="${MEDICARE_CONTAINER:-medicare-training}"
RESERVE_CONTAINER="${RESERVE_CONTAINER:-reserve-training}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Azure subscription"
az account show --query "{name:name, id:id}" -o table

echo ""
echo "==> Resource group: $RESOURCE_GROUP"
az group show --name "$RESOURCE_GROUP" -o table 2>/dev/null || {
  echo "ERROR: Resource group $RESOURCE_GROUP not found. Create it first or set RESOURCE_GROUP."
  exit 1
}

echo ""
echo "==> Storage account: $STORAGE_ACCOUNT"
if ! az storage account show --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
  if [[ "${CREATE_STORAGE_ACCOUNT:-false}" == "true" ]]; then
    LOCATION="${LOCATION:-$(az group show -n "$RESOURCE_GROUP" --query location -o tsv)}"
    az storage account create \
      --name "$STORAGE_ACCOUNT" \
      --resource-group "$RESOURCE_GROUP" \
      --location "$LOCATION" \
      --sku Standard_LRS \
      --kind StorageV2 \
      --enable-hierarchical-namespace false \
      --output none
    echo "    Created storage account $STORAGE_ACCOUNT"
  else
    echo "ERROR: Storage account $STORAGE_ACCOUNT not found in $RESOURCE_GROUP."
    echo "       Set STORAGE_ACCOUNT to an existing account, or CREATE_STORAGE_ACCOUNT=true to create one."
    exit 1
  fi
fi

echo ""
echo "==> Blob containers (Express Script landing zones)"
az storage container create \
  --account-name "$STORAGE_ACCOUNT" \
  --name "$MEDICARE_CONTAINER" \
  --auth-mode login \
  --public-access off \
  --output none 2>/dev/null || true
echo "    $MEDICARE_CONTAINER"

az storage container create \
  --account-name "$STORAGE_ACCOUNT" \
  --name "$RESERVE_CONTAINER" \
  --auth-mode login \
  --output none 2>/dev/null || true
echo "    $RESERVE_CONTAINER"

echo ""
echo "==> Folder layout (for Express Script team)"
echo "    abfss://$MEDICARE_CONTAINER@$STORAGE_ACCOUNT.dfs.core.windows.net/latest/"
echo "    abfss://$RESERVE_CONTAINER@$STORAGE_ACCOUNT.dfs.core.windows.net/latest/"
echo "    Expected files: claims.parquet or partitioned *.parquet"

echo ""
echo "==> Grant AML workspace identity read access on storage (RBAC)"
WS_PRINCIPAL_ID="$(az ml workspace show \
  --name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query identity.principalId -o tsv 2>/dev/null || true)"

if [[ -n "$WS_PRINCIPAL_ID" && "$WS_PRINCIPAL_ID" != "None" ]]; then
  STORAGE_ID="$(az storage account show \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --query id -o tsv)"
  az role assignment create \
    --assignee "$WS_PRINCIPAL_ID" \
    --role "Storage Blob Data Contributor" \
    --scope "$STORAGE_ID" \
    --output none 2>/dev/null \
    && echo "    Storage Blob Data Contributor → workspace MSI on $STORAGE_ACCOUNT" \
    || echo "    (role may already exist)"
else
  echo "    WARNING: Could not resolve workspace managed identity. Assign Storage Blob Data Contributor manually."
fi

echo ""
echo "==> Register AML datastores"
cd "$REPO_ROOT"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
pip install -q -r requirements-register.txt
TRAINING_STORAGE_ACCOUNT="$STORAGE_ACCOUNT" python scripts/register_training_datastore.py --both

echo ""
echo "==> Done"
echo ""
echo "Express Script should write to:"
echo "  https://$STORAGE_ACCOUNT.blob.core.windows.net/$MEDICARE_CONTAINER/latest/"
echo "  https://$STORAGE_ACCOUNT.blob.core.windows.net/$RESERVE_CONTAINER/latest/"
echo ""
echo "AML command job input (example):"
echo "  datastore: claims_medicare_training"
echo "  path: azureml://datastores/claims_medicare_training/paths/latest/"
echo ""
echo "Training script env:"
echo "  TRAINING_DATA_URI=/mnt/azureml/.../claims.parquet   (mounted path in job)"
echo "  or TRAINING_DATA_URI=wasbs://$MEDICARE_CONTAINER@$STORAGE_ACCOUNT.blob.core.windows.net/latest/claims.parquet"
