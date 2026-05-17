# Data

All raw datasets live in `data/raw/` and are **not committed to git** (see `.gitignore`).  
Processed artifacts in `data/processed/` are tracked by DVC.

---

## Elliptic Bitcoin Dataset (Graph branch)

**Source:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set  
**License:** Creative Commons Attribution 4.0  
**Size:** ~25 MB compressed

### Download (Kaggle CLI)

```bash
pip install kaggle
# Place your Kaggle API token at ~/.kaggle/kaggle.json
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/elliptic --unzip
```

Expected files after unzip:

```
data/raw/elliptic/
├── elliptic_txs_features.csv   # 203,769 rows × 167 cols (no header)
├── elliptic_txs_edgelist.csv   # 234,355 rows, cols: txId1,txId2
└── elliptic_txs_classes.csv    # 203,769 rows, cols: txId,class
```

---

## Synthetic Payment Memo Text (NLP branch)

Generated automatically by the pipeline — no download needed.

```bash
python -m multimodal_anti_money_laundering.data.make_dataset --generate-memos
```

Output: `data/processed/memo_texts.csv` (~50K rows, columns: txId, memo, label)

---

## CI / Offline Development

If the raw files are absent, `elliptic.py` falls back to a 10,000-node synthetic graph
so the full training pipeline (baseline + unit tests) runs without any downloads:

```bash
python -m multimodal_anti_money_laundering.train_model --model baseline --synthetic
```

---

## Best Practices

- Use **DVC** to version large data files instead of Git
- Track `.dvc` files in Git; store actual data remotely
- **Never commit** large data files directly to Git
