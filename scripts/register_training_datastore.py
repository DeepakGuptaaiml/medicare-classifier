"""
Register Azure ML datastores for Express Script training landing zones.

Creates (or updates) blob datastores pointing at medicare-training and
reserve-training containers so AML jobs can mount/read parquet snapshots.

Usage:
  az login
  export AZURE_CREDENTIALS='{"clientId":"...","clientSecret":"...","subscriptionId":"...","tenantId":"..."}'
  # or rely on DefaultAzureCredential after az login

  python scripts/register_training_datastore.py --storage-account claimsmlstorage4cd105406
  python scripts/register_training_datastore.py --storage-account claimsmlstorage4cd105406 --both
"""

from __future__ import annotations

import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore
from azure.identity import ClientSecretCredential, DefaultAzureCredential


RESOURCE_GROUP = "rg-claims-intelligence"
WORKSPACE_NAME = "claims-ml-workspace"

DATASTORES = {
    "claims_medicare_training": {
        "container": "medicare-training",
        "description": "Express Script landing zone — Medicare classifier training parquet",
    },
    "claims_reserve_training": {
        "container": "reserve-training",
        "description": "Express Script landing zone — Reserve forecaster training parquet",
    },
}


def get_ml_client() -> MLClient:
    azure_credentials = os.getenv("AZURE_CREDENTIALS")
    if azure_credentials:
        creds = json.loads(azure_credentials)
        credential = ClientSecretCredential(
            tenant_id=creds["tenantId"],
            client_id=creds["clientId"],
            client_secret=creds["clientSecret"],
        )
        subscription_id = creds["subscriptionId"]
    else:
        credential = DefaultAzureCredential()
        subscription_id = os.getenv(
            "AZURE_SUBSCRIPTION_ID",
            os.popen("az account show --query id -o tsv").read().strip(),
        )

    if not subscription_id:
        raise RuntimeError("Set AZURE_SUBSCRIPTION_ID or run az login")

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def register_datastore(
    client: MLClient,
    name: str,
    storage_account: str,
    container: str,
    description: str,
    *,
    account_key: str | None = None,
) -> None:
    kwargs: dict = {
        "name": name,
        "description": description,
        "account_name": storage_account,
        "container_name": container,
    }
    if account_key:
        kwargs["account_key"] = account_key

    datastore = AzureBlobDatastore(**kwargs)
    registered = client.datastores.create_or_update(datastore)
    print(f"Registered datastore: {registered.name}")
    print(f"  container: {container}")
    print(f"  account:   {storage_account}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register AML training datastores")
    parser.add_argument(
        "--storage-account",
        default=os.getenv("TRAINING_STORAGE_ACCOUNT", "claimsmlstorage4cd105406"),
        help="Storage account where Express Script writes parquet",
    )
    parser.add_argument(
        "--account-key",
        default=os.getenv("TRAINING_STORAGE_KEY"),
        help="Optional storage account key (omit if using workspace MSI/RBAC)",
    )
    parser.add_argument(
        "--medicare-only",
        action="store_true",
        help="Register only claims_medicare_training",
    )
    parser.add_argument(
        "--reserve-only",
        action="store_true",
        help="Register only claims_reserve_training",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Register both datastores (default when no filter flags)",
    )
    args = parser.parse_args()

    register_both = args.both or (not args.medicare_only and not args.reserve_only)
    client = get_ml_client()

    if args.medicare_only or register_both:
        meta = DATASTORES["claims_medicare_training"]
        register_datastore(
            client,
            "claims_medicare_training",
            args.storage_account,
            meta["container"],
            meta["description"],
            account_key=args.account_key,
        )

    if args.reserve_only or register_both:
        meta = DATASTORES["claims_reserve_training"]
        register_datastore(
            client,
            "claims_reserve_training",
            args.storage_account,
            meta["container"],
            meta["description"],
            account_key=args.account_key,
        )


if __name__ == "__main__":
    main()
