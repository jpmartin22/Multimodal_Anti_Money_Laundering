"""CLI entrypoint for daily Evidently drift monitoring.

Usage:
    # Run all three modality reports with defaults
    python -m multimodal_anti_money_laundering.monitoring.run_drift

    # Run a specific modality only
    python -m multimodal_anti_money_laundering.monitoring.run_drift --modality graph
    python -m multimodal_anti_money_laundering.monitoring.run_drift --modality timeseries
    python -m multimodal_anti_money_laundering.monitoring.run_drift --modality text

Reports are saved to reports/drift/*.html
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from multimodal_anti_money_laundering.monitoring.drift_report import (
    graph_drift_report,
    run_all_drift_reports,
    text_drift_report,
    timeseries_drift_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default data paths — match DVC-tracked locations
DEFAULT_GRAPH_FEATURES  = Path("data/processed/graph_features.npy")
DEFAULT_SEQUENCES       = Path("data/processed/bilstm_sequences.npy")
DEFAULT_LABELS          = Path("data/processed/bilstm_labels.npy")
DEFAULT_MEMO_CSV        = Path("data/raw/memo_dataset.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evidently drift reports")
    parser.add_argument(
        "--modality",
        choices=["graph", "timeseries", "text", "all"],
        default="all",
        help="Which modality to run drift detection for (default: all)",
    )
    parser.add_argument("--graph-features", type=Path, default=DEFAULT_GRAPH_FEATURES)
    parser.add_argument("--sequences",      type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--labels",         type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--memo-csv",       type=Path, default=DEFAULT_MEMO_CSV)
    args = parser.parse_args()

    if args.modality == "all":
        reports = run_all_drift_reports(
            graph_features_path=args.graph_features,
            sequences_path=args.sequences,
            labels_path=args.labels,
            memo_csv_path=args.memo_csv,
        )
        logger.info("All drift reports complete:")
        for modality, path in reports.items():
            logger.info("  %-12s → %s", modality, path)

    elif args.modality == "graph":
        graph_drift_report(args.graph_features)

    elif args.modality == "timeseries":
        timeseries_drift_report(args.sequences, args.labels)

    elif args.modality == "text":
        text_drift_report(args.memo_csv)


if __name__ == "__main__":
    main()
