"""Member C: Great Expectations Pre-Training Data Gate Validation."""
import numpy as np
import pandas as pd
import great_expectations as ge
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

def run_pre_training_validation(graph_features: np.ndarray, bilstm_sequences: np.ndarray, labels: np.ndarray):
    logger.info("Initializing Member C Great Expectations Validation Gate...")
    
    # --- 1. Validate Labels {0, 1} and No NaNs ---
    # Convert labels array to a GE pandas dataset for rapid data-level assertion
    df_labels = ge.dataset.PandasDataset(pd.DataFrame({"label": labels}))
    
    # Rule: Labels must strictly be 0 or 1
    res_label_values = df_labels.expect_column_values_to_be_in_set("label", [0, 1])
    # Rule: No NaNs allowed in target labels
    res_label_nulls = df_labels.expect_column_values_to_not_be_null("label")
    
    # --- 2. Validate Graph Features Matrix Structural Shapes (N, 165) ---
    graph_shape = graph_features.shape
    has_165_features = (len(graph_shape) == 2 and graph_shape[1] == 165)
    has_graph_nans = np.isnan(graph_features).any()
    
    # --- 3. Validate BiLSTM Sequence Structural Shapes (N, 49, 165) ---
    bilstm_shape = bilstm_sequences.shape
    is_valid_bilstm_tensor = (
        len(bilstm_shape) == 3 and 
        bilstm_shape[1] == 49 and 
        bilstm_shape[2] == 165
    )
    has_bilstm_nans = np.isnan(bilstm_sequences).any()
    
    # --- Evaluation Matrix Checkpoint Gates ---
    all_passed = True
    
    print("\n" + "="*20 + " GREAT EXPECTATIONS VALIDATION REPORT " + "="*20)
    
    # Print status flags for the user and logs
    if res_label_values.success and res_label_nulls.success:
        print("✅ Labels Validation: PASSED (Contains only {0, 1} and no nulls)")
    else:
        print("❌ Labels Validation: FAILED (Invalid target characters or NaN values detected)")
        all_passed = False
        
    if has_165_features and not has_graph_nans:
        print(f"✅ Graph Features Validation: PASSED (Shape matches (N, 165): {graph_shape})")
    else:
        print(f"❌ Graph Features Validation: FAILED (Shape is {graph_shape} instead of (N, 165) or NaNs found)")
        all_passed = False
        
    if is_valid_bilstm_tensor and not has_bilstm_nans:
        print(f"✅ BiLSTM Sequence Validation: PASSED (Shape matches (N, 49, 165): {bilstm_shape})")
    else:
        print(f"❌ BiLSTM Sequence Validation: FAILED (Shape is {bilstm_shape} instead of (N, 49, 165) or NaNs found)")
        all_passed = False
        
    print("="*78 + "\n")
    
    if not all_passed:
        logger.error("Pre-training data failed structural expectations gates!")
        raise ValueError("Data pipeline quality check failed. Halting training loop initialization.")
        
    logger.info("All data structures cleared Great Expectations gates successfully.")
    return True

if __name__ == "__main__":
    # Local Smoke-Testing Mock Matrices to verify the data gate operates correctly
    logger.info("Executing local verification run with valid placeholder data dimensions...")
    
    mock_n_samples = 100
    mock_graph = np.zeros((mock_n_samples, 165))
    mock_bilstm = np.zeros((mock_n_samples, 49, 165))
    mock_labels = np.random.choice([0, 1], size=mock_n_samples)
    
    run_pre_training_validation(mock_graph, mock_bilstm, mock_labels)