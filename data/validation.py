import pandas as pd
import great_expectations as ge
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

def validate_transaction_data(df: pd.DataFrame) -> bool:
    """Validate incoming transaction data schemas and distributions using Great Expectations."""
    logger.info("Executing Great Expectations Data Quality Suite...")
    
    # Wrap the pandas DataFrame into a Great Expectations dataset
    ge_df = ge.from_pandas(df)
    
    # 1. Validate Schema Invariants (Columns must exist)
    res_columns = ge_df.expect_table_columns_to_match_set(
        column_set=["amount", "type", "timestamp", "is_fraud"],
        exact_match=False
    )
    
    # 2. Validate Null Validation Boundaries (e.g., amount cannot be null)
    res_null = ge_df.expect_column_values_to_not_be_null("amount")
    
    # 3. Validate Data Distribution / Value Bounds (e.g., amount must be positive)
    res_bounds = ge_df.expect_column_values_to_be_between("amount", min_value=0)
    
    # Evaluate if all expectations passed
    success = res_columns.success and res_null.success and res_bounds.success
    
    if not success:
        logger.error("Data Quality Gate FAILED: Structural anomalies or unexpected distributions detected!")
        return False
        
    logger.info("Data Quality Gate PASSED: Ingestion schema satisfies all validation rules.")
    return True