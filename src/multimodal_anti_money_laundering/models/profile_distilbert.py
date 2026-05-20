import cProfile
import pstats
import io
import torch
from multimodal_anti_money_laundering.models.train_distilbert import DistilBertEncoder
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

def profile_inference():
    logger.info("Initiating structural profiling suite for DistilBERT...")
    
    # Instantiate the target execution component
    model = DistilBertEncoder()
    model.eval()
    
    test_memos = ["Urgent transfer request from suspect entity outside geographic bounds."] * 5
    
    # Initialize cProfile module wrapper
    pr = cProfile.Profile()
    pr.enable()
    
    with torch.no_grad():
        _, embeddings = model(test_memos)
        
    pr.disable()
    
    # Format and dump profiling telemetry output cleanly into logs terminal
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15) # Show top 15 heavy matrix operation function calls
    
    print("\n" + "="*20 + "MODEL PROFILING RESULTS " + "="*20)
    print(s.getvalue())
    print("="*74 + "\n")
    logger.info("Profiling analysis tracking phase successfully finalized.")

if __name__ == "__main__":
    torch.set_num_threads(1)
    profile_inference()