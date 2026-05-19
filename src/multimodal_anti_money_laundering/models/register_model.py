import mlflow
from mlflow.tracking import MlflowClient
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

def promote_best_model(experiment_name: str = "aml-baseline", model_name: str = "Multimodal_AML_Model"):
    """Find the latest run from an experiment, register it, and promote it to Staging."""
    logger.info(f"Connecting to MLflow tracking server to scan experiment: '{experiment_name}'")
    
    client = MlflowClient()
    
    # Fetch the experiment metadata
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.error(f"Registry Error: Experiment '{experiment_name}' not found.")
        return
        
    # Search for the latest run that successfully finished
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )
    
    if not runs:
        logger.error("Registry Error: No active tracking runs found to register.")
        return
        
    latest_run_id = runs[0].info.run_id
    logger.info(f"Identified top candidate run target: {latest_run_id}")
    
    # --- FIXED LOGIC START ---
    # Scan the run data to see what artifact path your teammates used when logging
    tags = runs[0].data.tags
    
    # Check if a model flavor was logged and extract its path, fallback to common project variations
    artifact_path = "model" 
    
    # Let's inspect the artifacts to see if a custom baseline/xgboost name was used
    if "mlflow.log-model.history" in tags:
        import json
        try:
            history = json.loads(tags["mlflow.log-model.history"])
            if history:
                artifact_path = history[0].get("artifact_path", "model")
                logger.info(f"Dynamically detected logged model artifact path: '{artifact_path}'")
        except Exception:
            pass
            
    # Connect directly to the tracked URI path location
    model_uri = f"runs:/{latest_run_id}/{artifact_path}"
    # --- FIXED LOGIC END ---

    try:
        # Register the model inside the MLflow Model Registry central database
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Successfully registered model version: V{model_version.version}")
        
        # Transition the model status to STAGING so Member D can pick it up for SageMaker deployment
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )
        logger.info(f"Model version V{model_version.version} successfully promoted to STAGING.")
        
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Registry Error: The run {latest_run_id} does not contain any logged model binaries.")
        logger.error("Ensure that 'mlflow.xgboost.log_model()' or 'mlflow.pytorch.log_model()' is actively executing in your train script.")
if __name__ == "__main__":
    promote_best_model()