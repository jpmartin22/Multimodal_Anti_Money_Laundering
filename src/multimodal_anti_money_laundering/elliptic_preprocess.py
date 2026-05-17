"""
elliptic_preprocess.py
======================
Member B (Neha) + Member A (Jaya) — Week 1 Elliptic preprocessing.

Produces TWO outputs from the same three raw Elliptic CSV files:

  OUTPUT 1 — Graph modality (for Member A / GraphSAGE)
  -------------------------------------------------------
  data/processed/graph_features.npy    (N_labeled, 165)
  data/processed/graph_labels.npy      (N_labeled,)      0=licit 1=illicit
  data/processed/graph_edge_index.npy  (2, E)            PyG-style edge index
  data/processed/graph_node_ids.npy    (N_labeled,)      txId per node

  OUTPUT 2 — Behavioral time-series modality (for Member B / BiLSTM)
  -------------------------------------------------------
  data/processed/bilstm_sequences.npy  (N_labeled, 49, 165)
                                        node × time_step × features
                                        unknown steps filled with zeros
  data/processed/bilstm_labels.npy     (N_labeled,)
  data/processed/elliptic_stats.json   dataset stats for Great Expectations

About the Elliptic dataset
--------------------------
  elliptic_txs_features.csv  — NO header. Columns are:
                                  col 0  : txId
                                  col 1  : time_step  (1–49)
                                  col 2–166 : 165 transaction features
                                  (first 94 = local tx features,
                                   next 71  = aggregated neighborhood features)
  elliptic_txs_classes.csv   — header: txId, class
                                  class 1 = illicit
                                  class 2 = licit
                                  class unknown = unlabeled (excluded from training)
  elliptic_txs_edgelist.csv  — header: txId1, txId2 (directed edges)

Usage
-----
    python elliptic_preprocess.py
    python elliptic_preprocess.py --data_dir data/raw/elliptic_bitcoin_dataset
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_TIME_STEPS = 49      # Elliptic spans 49 time steps
N_FEATURES   = 165     # columns 2–166 in features file
RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW FILES
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(data_dir: str):
    print(f"\n{'='*58}")
    print(f"  STEP 1 — Loading raw Elliptic files")
    print(f"{'='*58}")

    feat_path  = os.path.join(data_dir, "elliptic_txs_features.csv")
    cls_path   = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edge_path  = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    for p in [feat_path, cls_path, edge_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                f"Expected inside: {data_dir}"
            )

    # Features — NO header in this file
    print("  Loading features (no header)...")
    df_feat = pd.read_csv(feat_path, header=None)
    # Rename for clarity
    df_feat.columns = (
        ["txId", "time_step"] +
        [f"f{i}" for i in range(N_FEATURES)]
    )
    print(f"  Features shape   : {df_feat.shape}  "
          f"(expected ~203K rows × 167 cols)")

    # Classes
    print("  Loading classes...")
    df_cls = pd.read_csv(cls_path)
    df_cls.columns = ["txId", "label"]
    class_counts = df_cls["label"].value_counts().to_dict()
    print(f"  Label distribution: {class_counts}")

    # Edges
    print("  Loading edges...")
    df_edge = pd.read_csv(edge_path)
    df_edge.columns = ["txId1", "txId2"]
    print(f"  Edges            : {len(df_edge):,}")

    return df_feat, df_cls, df_edge


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MERGE + LABEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_filter(df_feat, df_cls):
    print(f"\n{'='*58}")
    print(f"  STEP 2 — Merging features with labels")
    print(f"{'='*58}")

    df = df_feat.merge(df_cls, on="txId", how="left")
    df["label"] = df["label"].fillna("unknown")

    total      = len(df)
    n_illicit  = (df["label"] == "1").sum()
    n_licit    = (df["label"] == "2").sum()
    n_unknown  = (df["label"] == "unknown").sum()

    print(f"  Total nodes      : {total:>10,}")
    print(f"  Illicit (1)      : {n_illicit:>10,}  ({n_illicit/total*100:.1f}%)")
    print(f"  Licit   (2)      : {n_licit:>10,}  ({n_licit/total*100:.1f}%)")
    print(f"  Unknown          : {n_unknown:>10,}  ({n_unknown/total*100:.1f}%)")

    # Convert label: 1 → 1 (illicit), 2 → 0 (licit), unknown → -1
    label_map = {"1": 1, "2": 0, "unknown": -1}
    df["label_int"] = df["label"].map(label_map)

    # Labeled only (exclude unknown for supervised training)
    df_labeled = df[df["label_int"] != -1].copy().reset_index(drop=True)
    print(f"\n  Labeled nodes kept : {len(df_labeled):>8,}")
    print(f"  Illicit rate       : "
          f"{df_labeled['label_int'].mean()*100:.2f}%  "
          f"(~2% expected)")

    return df, df_labeled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SCALE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df_all, df_labeled):
    """
    Fit StandardScaler on ALL labeled nodes' features.
    Apply to all nodes (needed for behavioral sequences).
    """
    print(f"\n{'='*58}")
    print(f"  STEP 3 — Scaling features")
    print(f"{'='*58}")

    feat_cols = [f"f{i}" for i in range(N_FEATURES)]
    scaler    = StandardScaler()

    # Fit on labeled nodes only (no data leakage from unknown)
    scaler.fit(df_labeled[feat_cols].values)

    # Apply to all nodes
    df_all    = df_all.copy()
    df_labeled = df_labeled.copy()
    df_all[feat_cols]     = scaler.transform(df_all[feat_cols].values)
    df_labeled[feat_cols] = scaler.transform(df_labeled[feat_cols].values)

    print(f"  Scaler fit on {len(df_labeled):,} labeled nodes")
    print(f"  Applied to all {len(df_all):,} nodes")

    return df_all, df_labeled, scaler, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD GRAPH OUTPUTS (for Member A / GraphSAGE)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_outputs(df_labeled, df_edge, feat_cols):
    print(f"\n{'='*58}")
    print(f"  STEP 4 — Building graph outputs (Member A)")
    print(f"{'='*58}")

    # Node features + labels
    X = df_labeled[feat_cols].values.astype(np.float32)   # (N, 165)
    y = df_labeled["label_int"].values.astype(np.int64)    # (N,)
    node_ids = df_labeled["txId"].values                   # (N,)

    # Build edge index (PyG format: shape [2, E])
    # Only keep edges where BOTH endpoints are labeled nodes
    labeled_set = set(node_ids.tolist())
    edges_filtered = df_edge[
        df_edge["txId1"].isin(labeled_set) &
        df_edge["txId2"].isin(labeled_set)
    ]

    # Map txId → integer index
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(node_ids)}
    src = edges_filtered["txId1"].map(id_to_idx).values
    dst = edges_filtered["txId2"].map(id_to_idx).values
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)  # (2, E)

    print(f"  Node features shape : {X.shape}")
    print(f"  Labels shape        : {y.shape}")
    print(f"  Edge index shape    : {edge_index.shape}")
    print(f"  Illicit nodes       : {y.sum():,}  ({y.mean()*100:.2f}%)")

    return X, y, edge_index, node_ids


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD BEHAVIORAL SEQUENCES (for Member B / BiLSTM)
# ─────────────────────────────────────────────────────────────────────────────

def build_bilstm_sequences(df_all, df_labeled, feat_cols):
    """
    For each labeled node, build a sequence of length 49
    (one entry per Elliptic time step).

    Since a node only exists at ONE time step, the sequence is:
        - zeros for all time steps before the node's step
        - the node's own features at its actual time step
        - zeros for all time steps after

    This encodes WHEN in the transaction lifecycle the node appeared,
    which is a genuine AML signal (illicit txs cluster in certain time windows).

    Shape: (N_labeled, 49, 165)
    """
    print(f"\n{'='*58}")
    print(f"  STEP 5 — Building BiLSTM sequences (Member B)")
    print(f"{'='*58}")

    feat_cols_arr = [f"f{i}" for i in range(N_FEATURES)]
    N = len(df_labeled)

    # Pre-allocate — zeros everywhere
    sequences = np.zeros((N, N_TIME_STEPS, N_FEATURES), dtype=np.float32)

    # Fill in each node's features at its own time step
    for i, (_, row) in enumerate(df_labeled.iterrows()):
        step_idx = int(row["time_step"]) - 1   # 0-indexed
        sequences[i, step_idx, :] = row[feat_cols_arr].values.astype(np.float32)

    labels = df_labeled["label_int"].values.astype(np.int64)

    print(f"  Sequences shape  : {sequences.shape}  "
          f"(nodes × time_steps × features)")
    print(f"  Labels shape     : {labels.shape}")
    print(f"  Non-zero steps   : each node has exactly 1 active time step")
    print(f"  Illicit rate     : {labels.mean()*100:.2f}%")
    print(f"\n  Time step distribution of labeled nodes:")
    step_counts = df_labeled["time_step"].value_counts().sort_index()
    print(f"    Min step : {step_counts.index.min()}")
    print(f"    Max step : {step_counts.index.max()}")
    print(f"    Avg nodes/step: {step_counts.mean():.0f}")

    return sequences, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    output_dir, scaler, feat_cols,
    graph_X, graph_y, edge_index, node_ids,
    bilstm_X, bilstm_y,
    df_labeled,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Graph outputs ─────────────────────────────────────────────────────────
    np.save(os.path.join(output_dir, "graph_features.npy"),   graph_X)
    np.save(os.path.join(output_dir, "graph_labels.npy"),     graph_y)
    np.save(os.path.join(output_dir, "graph_edge_index.npy"), edge_index)
    np.save(os.path.join(output_dir, "graph_node_ids.npy"),   node_ids)

    # ── BiLSTM outputs ────────────────────────────────────────────────────────
    np.save(os.path.join(output_dir, "bilstm_sequences.npy"), bilstm_X)
    np.save(os.path.join(output_dir, "bilstm_labels.npy"),    bilstm_y)

    # ── Stats JSON ────────────────────────────────────────────────────────────
    stats = {
        "n_labeled_nodes":     int(len(graph_y)),
        "n_illicit":           int(graph_y.sum()),
        "n_licit":             int((graph_y == 0).sum()),
        "illicit_rate":        round(float(graph_y.mean()), 6),
        "n_edges":             int(edge_index.shape[1]),
        "n_time_steps":        N_TIME_STEPS,
        "n_features":          N_FEATURES,
        "bilstm_shape":        list(bilstm_X.shape),
        "graph_shape":         list(graph_X.shape),
        "time_step_range":     [
            int(df_labeled["time_step"].min()),
            int(df_labeled["time_step"].max())
        ],
        "scaler_mean_sample":  scaler.mean_[:5].tolist(),
        "scaler_scale_sample": scaler.scale_[:5].tolist(),
        "train_val_test_split": "70 / 15 / 15 on labeled nodes",
    }
    stats_path = os.path.join(output_dir, "elliptic_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  ALL OUTPUTS SAVED → {output_dir}/")
    print(f"{'='*58}")
    print(f"\n  FOR MEMBER A (GraphSAGE):")
    print(f"    graph_features.npy   : {graph_X.shape}")
    print(f"    graph_labels.npy     : {graph_y.shape}")
    print(f"    graph_edge_index.npy : {edge_index.shape}")
    print(f"    graph_node_ids.npy   : {node_ids.shape}")
    print(f"\n  FOR MEMBER B (BiLSTM):")
    print(f"    bilstm_sequences.npy : {bilstm_X.shape}")
    print(f"    bilstm_labels.npy    : {bilstm_y.shape}")
    print(f"\n  SHARED:")
    print(f"    elliptic_stats.json  : dataset stats")
    print(f"\n  Illicit rate : {graph_y.mean()*100:.2f}%")
    print(f"  Edges kept   : {edge_index.shape[1]:,} "
          f"(both endpoints labeled)")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Elliptic preprocessing — graph + BiLSTM outputs."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="data/raw/elliptic_bitcoin_dataset",
        help="Folder containing the 3 Elliptic CSV files"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/processed",
        help="Output directory (default: data/processed)"
    )
    args = parser.parse_args()

    # Load
    df_feat, df_cls, df_edge = load_raw(args.data_dir)

    # Merge + filter
    df_all, df_labeled = merge_and_filter(df_feat, df_cls)

    # Scale
    df_all, df_labeled, scaler, feat_cols = scale_features(df_all, df_labeled)

    # Graph outputs
    graph_X, graph_y, edge_index, node_ids = build_graph_outputs(
        df_labeled, df_edge, feat_cols
    )

    # BiLSTM sequences
    bilstm_X, bilstm_y = build_bilstm_sequences(df_all, df_labeled, feat_cols)

    # Save everything
    save_outputs(
        args.output_dir, scaler, feat_cols,
        graph_X, graph_y, edge_index, node_ids,
        bilstm_X, bilstm_y,
        df_labeled,
    )


if __name__ == "__main__":
    main()
