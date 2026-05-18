# Inside make_dataset.py, update your process_data function:
from multimodal_anti_money_laundering.data.validation import validate_transaction_data

def process_data(input_dir: Path, output_dir: Path) -> None:
    logger.info("Reading raw data from %s", input_dir)
    
    # Example placeholder loading for your validation check
    # raw_df = pd.read_csv(input_dir / "transactions.csv")
    # if not validate_transaction_data(raw_df):
    #     raise ValueError("Pipeline stopped: Data quality metrics failed.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing processed data to %s", output_dir)