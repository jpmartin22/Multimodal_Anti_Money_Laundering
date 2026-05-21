import cProfile
import io
import pstats
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models/distilbert/memo_model")
BASE_MODEL = "distilbert-base-uncased"


def profile_inference():
    logger.info("Initiating structural profiling suite for DistilBERT...")

    model_path = MODEL_DIR if MODEL_DIR.exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()

    test_memos = [
        "Urgent transfer request from suspect entity outside geographic bounds."
    ] * 5

    pr = cProfile.Profile()
    pr.enable()

    with torch.no_grad():
        inputs = tokenizer(
            test_memos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        inputs.pop("token_type_ids", None)
        outputs = model(**inputs)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)

    print("\n" + "=" * 20 + "MODEL PROFILING RESULTS " + "=" * 20)
    print(f"Model source: {model_path}")
    print(f"Logits shape: {tuple(outputs.logits.shape)}")
    print(s.getvalue())
    print("=" * 74 + "\n")
    logger.info("Profiling analysis tracking phase successfully finalized.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    profile_inference()
