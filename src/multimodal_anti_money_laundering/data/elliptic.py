"""Elliptic Bitcoin dataset loader and PyG graph builder.

Expected raw files under data/raw/elliptic/:
  elliptic_txs_features.csv  — no header; col 0 = txId, cols 1-166 = features
  elliptic_txs_edgelist.csv  — header: txId1, txId2
  elliptic_txs_classes.csv   — header: txId, class  (1=illicit, 2=licit, unknown)

If the raw files are absent, ``load_or_build_graph`` falls back to a
deterministic synthetic graph of the same shape so CI smoke tests pass
without the Kaggle download.

Download instructions: see data/README.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from multimodal_anti_money_laundering.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)

ELLIPTIC_RAW_DIR: Path = RAW_DATA_DIR / "elliptic"
N_NODE_FEATURES: int = 166  # time-step column is feature_0; 165 + 1 = 166 total


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------


def _load_raw_files(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three Elliptic CSV files from *data_dir*.

    Returns:
        features_df: (n_nodes, 167) — txId + 166 features, no header in file.
        edges_df:    (n_edges, 2)  — txId1, txId2.
        classes_df:  (n_nodes, 2) — txId, class.
    """
    features_path = data_dir / "elliptic_txs_features.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"

    for p in (features_path, edges_path, classes_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Elliptic file: {p}\n"
                "Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
                "and place the three CSVs in data/raw/elliptic/"
            )

    feature_cols = ["txId"] + [f"f{i}" for i in range(N_NODE_FEATURES)]
    features_df = pd.read_csv(features_path, header=None, names=feature_cols)
    edges_df = pd.read_csv(edges_path)
    classes_df = pd.read_csv(classes_path)

    logger.info(
        "Elliptic raw: %d transactions, %d edges", len(features_df), len(edges_df)
    )
    return features_df, edges_df, classes_df


# ---------------------------------------------------------------------------
# Synthetic fallback (for CI / offline development)
# ---------------------------------------------------------------------------


def _make_synthetic(
    n_nodes: int = 10_000,
    n_edges: int = 15_000,
    illicit_frac: float = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return synthetic dataframes with the same schema as the real Elliptic files.

    Illicit nodes have a slight positive mean shift on the first 10 features
    so a linear classifier has a non-trivial signal to learn.
    """
    rng = np.random.default_rng(seed)

    tx_ids = np.arange(n_nodes)
    n_illicit = int(n_nodes * illicit_frac)
    n_licit = n_nodes - n_illicit

    # Features: illicit nodes shifted by +0.5 on first 10 dims
    feats_licit = rng.normal(0, 1, (n_licit, N_NODE_FEATURES)).astype(np.float32)
    feats_illicit = rng.normal(0, 1, (n_illicit, N_NODE_FEATURES)).astype(np.float32)
    feats_illicit[:, :10] += 0.5

    all_feats = np.vstack([feats_licit, feats_illicit])
    all_ids = np.concatenate([tx_ids[:n_licit], tx_ids[n_licit:]])

    # Shuffle so illicit nodes are not all at the end
    perm = rng.permutation(n_nodes)
    all_ids = all_ids[perm]
    all_feats = all_feats[perm]

    feature_cols = ["txId"] + [f"f{i}" for i in range(N_NODE_FEATURES)]
    features_df = pd.DataFrame(
        np.column_stack([all_ids, all_feats]), columns=feature_cols
    )
    features_df["txId"] = features_df["txId"].astype(int)

    # Edges: random pairs
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    mask = src != dst
    edges_df = pd.DataFrame({"txId1": src[mask], "txId2": dst[mask]})

    # Labels: first n_licit of the shuffled array → licit (class=2), rest → illicit (class=1)
    # Reconstruct from perm
    orig_label = np.where(perm < n_licit, 2, 1)  # 2=licit, 1=illicit
    classes_df = pd.DataFrame({"txId": all_ids, "class": orig_label})

    logger.info(
        "Synthetic Elliptic: %d nodes (%d illicit), %d edges",
        n_nodes,
        n_illicit,
        len(edges_df),
    )
    return features_df, edges_df, classes_df


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_pyg_graph(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    normalize: bool = True,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> "torch_geometric.data.Data":  # type: ignore[name-defined]
    """Build a PyTorch Geometric ``Data`` object from the Elliptic dataframes.

    Only labeled nodes (class ∈ {1, 2}) appear in train/val/test masks.
    Unknown nodes contribute edges and features but are excluded from loss.

    Args:
        features_df: txId + 166 feature columns.
        edges_df: Directed edge list (txId1, txId2).
        classes_df: Per-transaction labels (txId, class).
        normalize: Apply ``StandardScaler`` fitted on training nodes.
        train_frac: Fraction of labeled nodes for training.
        val_frac: Fraction of labeled nodes for validation.
        seed: Random seed for split reproducibility.

    Returns:
        PyG ``Data`` with x, edge_index, y, train_mask, val_mask, test_mask.
    """
    from torch_geometric.data import Data  # deferred so module imports without PyG

    tx_ids = features_df["txId"].values.astype(int)
    id_to_idx: dict[int, int] = {int(tid): i for i, tid in enumerate(tx_ids)}
    n_nodes = len(tx_ids)

    # ---- labels ----
    label_map = {1: 1, 2: 0, "1": 1, "2": 0, "unknown": -1}
    classes_df = classes_df.copy()
    classes_df["label"] = classes_df["class"].map(label_map).fillna(-1).astype(int)
    label_lookup = classes_df.set_index("txId")["label"].to_dict()
    y_raw = np.array([label_lookup.get(int(tid), -1) for tid in tx_ids], dtype=np.int64)

    # ---- node features ----
    feat_cols = [c for c in features_df.columns if c != "txId"]
    x_np = features_df[feat_cols].values.astype(np.float32)

    # ---- train/val/test masks (labeled nodes only) ----
    labeled_idx = np.where(y_raw != -1)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(labeled_idx)
    n_labeled = len(labeled_idx)
    n_train = int(train_frac * n_labeled)
    n_val = int(val_frac * n_labeled)

    train_idx = labeled_idx[:n_train]
    val_idx = labeled_idx[n_train : n_train + n_val]
    test_idx = labeled_idx[n_train + n_val :]

    if normalize:
        scaler = StandardScaler()
        x_np[train_idx] = scaler.fit_transform(x_np[train_idx])
        x_np[val_idx] = scaler.transform(x_np[val_idx])
        x_np[test_idx] = scaler.transform(x_np[test_idx])

    import torch  # deferred — tabular path works without torch

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ---- edges (filter to known nodes) ----
    src_ids = edges_df["txId1"].astype(int).values
    dst_ids = edges_df["txId2"].astype(int).values
    valid = np.array(
        [s in id_to_idx and d in id_to_idx for s, d in zip(src_ids, dst_ids)]
    )
    src_idx = np.array([id_to_idx[int(s)] for s in src_ids[valid]])
    dst_idx = np.array([id_to_idx[int(d)] for d in dst_ids[valid]])
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    data = Data(
        x=torch.tensor(x_np, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_raw, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=n_nodes,
    )

    n_illicit = int((y_raw == 1).sum())
    n_licit = int((y_raw == 0).sum())
    n_unknown = int((y_raw == -1).sum())
    logger.info(
        "PyG graph: %d nodes | illicit=%d (%.1f%%) licit=%d unknown=%d | %d edges",
        n_nodes,
        n_illicit,
        100.0 * n_illicit / max(n_illicit + n_licit, 1),
        n_licit,
        n_unknown,
        data.num_edges,
    )
    return data


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def load_or_build_graph(
    processed_path: Optional[Path] = None,
    force_rebuild: bool = False,
    use_synthetic: bool = False,
) -> "torch_geometric.data.Data":  # type: ignore[name-defined]
    """Return a cached PyG graph, building it from raw CSVs (or synthetic data) if needed.

    Args:
        processed_path: ``.pt`` file to cache/load the built graph.
        force_rebuild: Rebuild even if the cache file exists.
        use_synthetic: Skip raw files and use synthetic data (for testing).

    Returns:
        PyG ``Data`` object ready for GraphSAGE training.
    """
    if processed_path is None:
        processed_path = PROCESSED_DATA_DIR / "elliptic_graph.pt"

    if processed_path.exists() and not force_rebuild and not use_synthetic:
        import torch  # deferred
        logger.info("Loading cached graph from %s", processed_path)
        return torch.load(str(processed_path), weights_only=False)

    if use_synthetic or not (ELLIPTIC_RAW_DIR / "elliptic_txs_features.csv").exists():
        logger.warning(
            "Raw Elliptic files not found in %s — using synthetic fallback.",
            ELLIPTIC_RAW_DIR,
        )
        features_df, edges_df, classes_df = _make_synthetic()
    else:
        features_df, edges_df, classes_df = _load_raw_files(ELLIPTIC_RAW_DIR)

    data = build_pyg_graph(features_df, edges_df, classes_df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    import torch  # deferred
    torch.save(data, str(processed_path))
    logger.info("Saved processed graph to %s", processed_path)
    return data


def load_tabular(
    use_synthetic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) as NumPy arrays.

    Loads the Elliptic node features in tabular form (no graph structure) for
    the XGBoost baseline.  Labels are binary: 1=illicit, 0=licit.
    Unknown-labeled rows are excluded.
    """
    if use_synthetic or not (ELLIPTIC_RAW_DIR / "elliptic_txs_features.csv").exists():
        logger.warning("Using synthetic tabular data for baseline.")
        features_df, _, classes_df = _make_synthetic()
    else:
        features_df, _, classes_df = _load_raw_files(ELLIPTIC_RAW_DIR)

    label_map = {1: 1, 2: 0, "1": 1, "2": 0}
    classes_df = classes_df.copy()
    classes_df["label"] = classes_df["class"].map(label_map)
    merged = features_df.merge(classes_df[["txId", "label"]], on="txId")
    labeled = merged.dropna(subset=["label"])
    labeled = labeled.copy()
    labeled["label"] = labeled["label"].astype(int)

    feat_cols = [c for c in labeled.columns if c not in ("txId", "label")]
    X = labeled[feat_cols].values.astype(np.float32)
    y = labeled["label"].values.astype(np.int32)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n = len(idx)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_i = idx[:n_train]
    val_i = idx[n_train : n_train + n_val]
    test_i = idx[n_train + n_val :]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_i])
    X_val = scaler.transform(X[val_i])
    X_test = scaler.transform(X[test_i])

    logger.info(
        "Tabular split: train=%d val=%d test=%d | illicit train=%.2f%%",
        len(train_i),
        len(val_i),
        len(test_i),
        100.0 * y[train_i].mean(),
    )
    return X_train, y[train_i], X_val, y[val_i], X_test, y[test_i]
