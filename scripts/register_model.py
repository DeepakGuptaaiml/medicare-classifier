"""
Register ML model in Azure ML model registry.
Run after training or as part of CD pipeline.
"""

import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential, DefaultAzureCredential


def get_ml_client():
    """
    Connect to Azure ML workspace.
    CI/CD: reads AZURE_CREDENTIALS JSON from environment.
    Local: falls back to DefaultAzureCredential (az login).
    Credentials NEVER hardcoded — always from environment.
    """
    azure_credentials = os.getenv("AZURE_CREDENTIALS")

    if azure_credentials:
        creds = json.loads(azure_credentials)
        credential = ClientSecretCredential(
            tenant_id=creds["tenantId"],
            client_id=creds["clientId"],
            client_secret=creds["clientSecret"],
        )
        subscription_id = creds.get(
            "subscriptionId",
            "c60ebf0e-9a32-46b8-9efa-8dc1e2b2cddc",
        )
    else:
        credential = DefaultAzureCredential()
        subscription_id = "c60ebf0e-9a32-46b8-9efa-8dc1e2b2cddc"

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name="rg-claims-intelligence",
        workspace_name="claims-ml-workspace",
    )


def register_model(
    model_path: str,
    model_name: str,
    description: str,
    tags: dict,
):
    """Register model artifact in Azure ML registry."""
    client = get_ml_client()

    model = Model(
        path=model_path,
        name=model_name,
        description=description,
        type=AssetTypes.CUSTOM_MODEL,
        tags={k: str(v) for k, v in tags.items()},
    )

    registered = client.models.create_or_update(model)
    print(f"Registered: {registered.name} v{registered.version}")
    print(f"Path: {registered.path}")
    print(f"Tags: {registered.tags}")
    return registered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["medicare", "reserve"],
        required=True,
    )
    parser.add_argument(
        "--commit-sha",
        default="local",
    )
    args = parser.parse_args()

    if args.model == "medicare":
        register_model(
            model_path="models/medicare_classifier.pkl",
            model_name="medicare-classifier",
            description=(
                "AdaBoost classifier for MMSEA Section 111 Medicare "
                "claim identification. Predicts is_medicare_reportable "
                "based on 20 features including pay codes, ORM/TPOC "
                "thresholds, and claimant eligibility rules."
            ),
            tags={
                "algorithm": "AdaBoost",
                "sampling_strategy": "undersampled",
                "recall": "1.0",
                "f1_score": "0.64",
                "roc_auc": "0.82",
                "accuracy": "0.63",
                "target": "is_medicare_reportable",
                "feature_count": "20",
                "problem_type": "classification",
                "regulation": "MMSEA_Section_111",
                "environment": "production",
                "commit_sha": args.commit_sha,
                "framework": "scikit-learn",
                "data_source": "WC_Medicare_Claims",
            },
        )

    elif args.model == "reserve":
        register_model(
            model_path="models/best_reserve_model.pkl",
            model_name="reserve-forecaster",
            description=(
                "XGBoost regressor for WC Medical-Only claim reserve "
                "forecasting. Predicts total medical reserve (reserve_3) "
                "from 12 claim intake features including cause codes, "
                "body part, state of jurisdiction, and CMS report date."
            ),
            tags={
                "algorithm": "XGBoost",
                "problem_type": "regression",
                "target": "reserve_3_medical_reserve",
                "feature_count": "12",
                "claim_type": "WC_Medical_Only",
                "environment": "production",
                "commit_sha": args.commit_sha,
                "framework": "xgboost",
                "data_source": "WC_Claims_Reserve",
            },
        )
