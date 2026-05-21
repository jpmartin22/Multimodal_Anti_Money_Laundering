"""Member C: Automated AUC-PR Evaluation Gate for CI/CD Pipelines."""
import json
import sys
import logging  # Added standard library import
from pathlib import Path
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

# Hardcoded threshold defined in the project success metrics
TARGET_AUC_PR = 0.80

def check_metrics_gate(metrics_json_path: str = "reports/metrics.json") -> None:
    """Evaluate if the trained model satisfies regulatory performance thresholds."""
    logger.info("Starting automated Member C evaluation gate checks...")
    
    path = Path(metrics_json_path)
    if not path.exists():
        logger.error(f"Evaluation Gate Failed: Metrics file '{metrics_json_path}' not found.")
        sys.exit(1)
        
    with open(path, "r") as f:
        metrics = json.load(f)
        
    # Extract the primary metric
    auc_pr = metrics.get("auc_pr", 0.0)
    logger.info(f"Evaluated Model AUC-PR: {auc_pr:.4f} (Target: >= {TARGET_AUC_PR})")
    
    if auc_pr < TARGET_AUC_PR:
        logger.error(f"CRITICAL CRASH: Model performance ({auc_pr:.4f}) dropped below regulatory compliance target ({TARGET_AUC_PR})!")
        sys.exit(1)
        
    logger.info("SUCCESS: Model passed the evaluation gate. Proceeding to registration pipeline.")
    sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/metrics.json")
    args = parser.parse_args()
    
    # FIX: Force Python to format standalone terminal runs to look exactly like the project style
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    print(">>> Python execution reached __main__ block successfully!")
    check_metrics_gate(args.metrics)