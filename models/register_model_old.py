"""Member C: Automated MLflow Model Registry Promotion Logic."""
import mlflow
from mlflow.tracking import MlflowClient
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

def promote_best_model(run_id: str, model_name: str = "multimodal_aml_network"):
    """Register a passing model artifact and promote it automatically to Staging."""
    logger.info(f"Connecting to MLflow Model Registry to promote run: {run_id}")
    
    client = MlflowClient()
    
    # 1. Register the model artifact
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    logger.info(f"Model successfully registered as '{model_name}' Version {mv.version}")
    
    # 2. Transition its stage automatically to Staging so Member D can containerize it
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=True
    )
    logger.info(f"Version {mv.version} promoted to STAGING stage.")

if __name__ == "__main__":
    logger.info("MLflow model registry management utilities initialized.")