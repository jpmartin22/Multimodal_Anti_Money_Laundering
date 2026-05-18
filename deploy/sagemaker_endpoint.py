"""Create or update the AML scorer SageMaker endpoint.

Usage:
    # Deploy (or update) with canary rollout config
    python deploy/sagemaker_endpoint.py --config deploy/canary_rollout.yaml

    # Promote canary to 100% after validation
    python deploy/sagemaker_endpoint.py --config deploy/canary_rollout.yaml --promote

    # Delete the endpoint entirely
    python deploy/sagemaker_endpoint.py --config deploy/canary_rollout.yaml --delete

Prerequisites:
    pip install boto3 sagemaker pyyaml
    AWS credentials configured (env vars or ~/.aws/credentials)
    IAM role with AmazonSageMakerFullAccess + AmazonEC2ContainerRegistryReadOnly
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import boto3
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_config(config_path: Path) -> dict:
    import os
    with open(config_path) as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)  # resolves ${AWS_ACCOUNT_ID} from env
    return yaml.safe_load(expanded)


def get_execution_role(region: str) -> str:
    """Return the SageMaker execution role ARN from environment or SSM."""
    import os
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        raise EnvironmentError(
            "Set SAGEMAKER_ROLE_ARN environment variable to your SageMaker IAM role ARN.\n"
            "Example: arn:aws:iam::<account>:role/SageMakerExecutionRole"
        )
    return role


def register_model(sm: object, variant: dict, image_uri: str, role_arn: str) -> str:
    """Create a SageMaker model for the given variant. Returns the model name."""
    model_name = variant["model_name"]
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": image_uri, "Mode": "SingleModel"},
            ExecutionRoleArn=role_arn,
        )
        logger.info("Registered model: %s", model_name)
    except sm.exceptions.ResourceInUse:
        logger.info("Model already exists, skipping: %s", model_name)
    return model_name


def create_endpoint_config(sm: object, cfg: dict, role_arn: str) -> str:
    """Create a SageMaker endpoint config with canary variants. Returns config name."""
    config_name = f"{cfg['endpoint_name']}-canary-config"

    production_variants = []
    image_uris = [cfg["image_uri"], cfg["canary_image_uri"]]

    for variant, image_uri in zip(cfg["production_variants"], image_uris):
        register_model(sm, variant, image_uri, role_arn)
        production_variants.append({
            "VariantName": variant["variant_name"],
            "ModelName": variant["model_name"],
            "InitialInstanceCount": variant["initial_instance_count"],
            "InstanceType": variant["instance_type"],
            "InitialVariantWeight": variant["initial_weight"],
        })

    try:
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=production_variants,
        )
        logger.info("Created endpoint config: %s", config_name)
    except sm.exceptions.ResourceInUse:
        logger.info("Endpoint config already exists: %s", config_name)

    return config_name


def deploy(cfg: dict) -> None:
    region = cfg["region"]
    sm = boto3.client("sagemaker", region_name=region)
    role_arn = get_execution_role(region)

    config_name = create_endpoint_config(sm, cfg, role_arn)
    endpoint_name = cfg["endpoint_name"]

    existing = [e["EndpointName"] for e in sm.list_endpoints()["Endpoints"]]
    if endpoint_name in existing:
        logger.info("Updating endpoint %s ...", endpoint_name)
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        logger.info("Creating endpoint %s ...", endpoint_name)
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    logger.info("Endpoint is live: %s", endpoint_name)


def promote(cfg: dict) -> None:
    """Shift 100% traffic to Primary variant (full rollout after canary passes)."""
    sm = boto3.client("sagemaker", region_name=cfg["region"])
    sm.update_endpoint_weights_and_capacities(
        EndpointName=cfg["endpoint_name"],
        DesiredWeightsAndCapacities=[
            {"VariantName": "Primary", "DesiredWeight": 100},
            {"VariantName": "Canary", "DesiredWeight": 0},
        ],
    )
    logger.info("Promoted Primary to 100%% traffic on %s", cfg["endpoint_name"])


def delete(cfg: dict) -> None:
    sm = boto3.client("sagemaker", region_name=cfg["region"])
    sm.delete_endpoint(EndpointName=cfg["endpoint_name"])
    logger.info("Deleted endpoint: %s", cfg["endpoint_name"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy AML scorer to SageMaker")
    parser.add_argument("--config", type=Path, default=Path("deploy/canary_rollout.yaml"))
    parser.add_argument("--promote", action="store_true",
                        help="Shift 100%% traffic to Primary variant")
    parser.add_argument("--delete", action="store_true",
                        help="Tear down the endpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.delete:
        delete(cfg)
    elif args.promote:
        promote(cfg)
    else:
        deploy(cfg)


if __name__ == "__main__":
    main()
