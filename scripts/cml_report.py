"""CML report generator — runs in GitHub Actions and produces reports/cml_report.md.

Reads stored metrics JSON files (no model training required) and generates:
  1. Markdown metrics table — all models vs proposal targets
  2. AUC-PR comparison bar chart  (reports/figures/cml_auc_pr.png)
  3. Multi-metric comparison chart (reports/figures/cml_metrics.png)
  4. Current vs. baseline delta table
  5. Full markdown report          (reports/cml_report.md)

The cml.yml workflow posts cml_report.md as a PR comment via iterative/cml.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Load metrics ─────────────────────────────────────────────────────────────


def load_metrics() -> dict:
    def _read(p: Path) -> dict:
        return json.loads(p.read_text()) if p.exists() else {}

    gs_raw = _read(ROOT / "graphsage_metrics.json")
    bl_raw = _read(ROOT / "models" / "baseline_metrics.json")
    bilstm = _read(ROOT / "bilstm_metrics.json")
    fusion = _read(ROOT / "fusion_metrics.json")
    gs_exp = _read(ROOT / "reports" / "graphsage_experiment_comparison.json")
    serving = _read(ROOT / "reports" / "profiling" / "serving_benchmark.json")
    snapshot = _read(ROOT / "reports" / "baseline_snapshot.json")

    gs_test = gs_raw.get("test_metrics", gs_raw)

    return {
        "baseline": {
            "label": "XGBoost Baseline",
            "auc_pr": bl_raw.get("auc_pr", 0),
            "prec_at_r80": bl_raw.get("precision_at_r80", 0),
            "fpr": bl_raw.get("fpr_at_r80", 1.0),
            "f1": bl_raw.get("f1", 0),
            "colour": "#d62728",
        },
        "graphsage": {
            "label": "GraphSAGE (best)",
            "auc_pr": gs_test.get("auc_pr", 0),
            "prec_at_r80": gs_test.get("prec_at_recall_80", 0),
            "fpr": gs_test.get("false_positive_rate", 1.0),
            "f1": gs_test.get("f1_fraud", 0),
            "colour": "#2ca02c",
        },
        "bilstm": {
            "label": "BiLSTM",
            "auc_pr": bilstm.get("auc_pr", 0),
            "prec_at_r80": bilstm.get("prec_at_recall_80", 0),
            "fpr": bilstm.get("false_positive_rate", 1.0),
            "f1": bilstm.get("f1_fraud", 0),
            "colour": "#9467bd",
        },
        "fusion": {
            "label": "Late-Fusion MLP",
            "auc_pr": fusion.get("auc_pr", 0),
            "prec_at_r80": fusion.get("prec_at_recall_80", 0),
            "fpr": fusion.get("false_positive_rate", 1.0),
            "f1": fusion.get("f1_fraud", 0),
            "colour": "#17becf",
        },
        "_serving": serving,
        "_gs_exp": gs_exp,
        "_snapshot": snapshot,
    }


# ── Chart 1: AUC-PR bar chart ────────────────────────────────────────────────


def plot_auc_pr(data: dict) -> Path:
    models = ["baseline", "graphsage", "bilstm", "fusion"]
    labels = [data[m]["label"] for m in models]
    values = [data[m]["auc_pr"] for m in models]
    colours = [data[m]["colour"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colours, alpha=0.85, zorder=3)
    ax.axhline(
        0.80,
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        label="Target AUC-PR = 0.80",
        zorder=4,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUC-PR", fontsize=11)
    ax.set_title("AUC-PR — All Models vs Target", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out = FIGURES / "cml_auc_pr.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Chart 2: Multi-metric grouped bar chart ───────────────────────────────────


def plot_multi_metric(data: dict) -> Path:
    models = ["baseline", "graphsage", "bilstm", "fusion"]
    labels = [data[m]["label"] for m in models]
    metrics = {
        "AUC-PR": [data[m]["auc_pr"] for m in models],
        "Prec@Recall=0.8": [data[m]["prec_at_r80"] for m in models],
        "F1 (fraud)": [data[m]["f1"] for m in models],
    }

    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, (metric, vals) in enumerate(metrics.items()):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=metric,
            color=palette[i],
            alpha=0.85,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Multi-Metric Comparison — AUC-PR · Prec@R=0.8 · F1",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out = FIGURES / "cml_metrics.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Delta comparison: current run vs committed baseline snapshot ──────────────


def build_delta_section(data: dict) -> list[str]:
    """Return markdown lines comparing current metrics to baseline_snapshot.json."""
    snap = data.get("_snapshot", {})
    if not snap or "_meta" not in snap:
        return []

    snap_date = snap.get("_meta", {}).get("created", "unknown")
    snap_desc = snap.get("_meta", {}).get("description", "")

    def _delta(current: float, previous: float, lower_is_better: bool = False) -> str:
        """Format a Δ cell: green ▲ for improvement, red ▼ for regression, — for no change."""
        if previous == 0 or current == previous:
            return "—"
        raw_diff = current - previous
        improvement = (-raw_diff) if lower_is_better else raw_diff
        symbol = "▲" if improvement > 0 else "▼"
        colour = "green" if improvement > 0 else "red"
        return f"<span style='color:{colour}'>{symbol} {abs(raw_diff):.4f}</span>"

    model_keys = [
        ("graphsage", "GraphSAGE"),
        ("bilstm", "BiLSTM"),
        ("fusion", "Late-Fusion MLP"),
    ]

    lines = [
        "### Current vs. Baseline Snapshot",
        "",
        f"> Baseline: `{snap_date}` — {snap_desc}",
        "",
        "| Model | Metric | Baseline | Current | Δ |",
        "|-------|--------|----------|---------|---|",
    ]

    for key, label in model_keys:
        snap_m = snap.get(key, {})
        curr_m = data.get(key, {})
        if not snap_m or not curr_m:
            continue

        metrics_cfg = [
            ("auc_pr", "AUC-PR", False),
            ("prec_at_r80", "Prec@Recall=0.8", False),
            ("fpr", "FPR", True),
            ("f1", "F1 (fraud)", False),
        ]
        snap_key_map = {
            "auc_pr": "auc_pr",
            "prec_at_r80": "prec_at_recall_80",
            "fpr": "false_positive_rate",
            "f1": "f1_fraud",
        }

        for curr_key, display, lower in metrics_cfg:
            s_key = snap_key_map[curr_key]
            s_val = snap_m.get(s_key, None)
            c_val = curr_m.get(curr_key, None)
            if s_val is None or c_val is None:
                continue
            delta = _delta(c_val, s_val, lower_is_better=lower)
            lines.append(
                f"| {label} | {display} | {s_val:.4f} | {c_val:.4f} | {delta} |"
            )

    lines.append("")
    return lines


# ── Build markdown report ─────────────────────────────────────────────────────


def build_report(data: dict, chart1: Path, chart2: Path) -> str:
    serving = data.get("_serving", {})
    p95 = serving.get("p95_ms", "N/A")
    tput = serving.get("throughput_rps", "N/A")
    sla = "✅" if serving.get("sla_passed") else "❌"

    gs = data["graphsage"]
    bl = data["baseline"]
    bi = data["bilstm"]
    fu = data["fusion"]

    def _pass(val, target, lower=False):
        ok = val <= target if lower else val >= target
        return f"**{val:.4f}** ✅" if ok else f"{val:.4f} ❌"

    lines = [
        "## 🤖 CML Model Report — Multimodal AML",
        "",
        "> Auto-generated by `scripts/cml_report.py` on every push/PR.",
        "",
        "### Success Metrics vs Targets",
        "",
        "| Metric | Target | XGBoost Baseline | GraphSAGE | BiLSTM | Late-Fusion MLP |",
        "|--------|--------|-----------------|-----------|--------|-----------------|",
        f"| AUC-PR (primary) | ≥ 0.80 | {_pass(bl['auc_pr'], 0.80)} | {_pass(gs['auc_pr'], 0.80)} | {_pass(bi['auc_pr'], 0.80)} | {_pass(fu['auc_pr'], 0.80)} |",
        f"| Prec @ Recall=0.8 | ≥ 0.70 | {_pass(bl['prec_at_r80'], 0.70)} | {_pass(gs['prec_at_r80'], 0.70)} | {_pass(bi['prec_at_r80'], 0.70)} | {_pass(fu['prec_at_r80'], 0.70)} |",
        f"| False Positive Rate | ≤ 0.05 | {_pass(bl['fpr'], 0.05, lower=True)} | {_pass(gs['fpr'], 0.05, lower=True)} | {_pass(bi['fpr'], 0.05, lower=True)} | {_pass(fu['fpr'], 0.05, lower=True)} |",
        f"| Inference P95 | < 200 ms | — | {p95} ms {sla} | — | — |",
        f"| Throughput | — | — | {tput} req/s | — | — |",
        "",
        "### AUC-PR Comparison",
        "",
        "![AUC-PR Chart](reports/figures/cml_auc_pr.png)",
        "",
        "### Multi-Metric Comparison",
        "",
        "![Multi-Metric Chart](reports/figures/cml_metrics.png)",
        "",
    ]

    lines += build_delta_section(data)

    lines += [
        "### GraphSAGE Experiment Runs",
        "",
        "| Run | lr | Hidden | Dropout | Val AUC-PR | Test AUC-PR | Time |",
        "|-----|----|--------|---------|------------|-------------|------|",
    ]

    gs_exp = data.get("_gs_exp", {})
    best = gs_exp.get("best_run", "")
    for exp in gs_exp.get("experiments", []):
        star = " ★" if exp["Run"] == best else ""
        lines.append(
            f"| `{exp['Run']}{star}` | {exp['lr']} | {exp['hidden_dim']} "
            f"| {exp['dropout']} | {exp['val_auc_pr']:.4f} "
            f"| {exp['test_auc_pr']:.4f} | {exp['train_min']} min |"
        )

    lines += [
        "",
        "> ★ Best run promoted to MLflow model registry.",
        "",
        "### Fusion Status",
        "",
        "| Component | Status |",
        "|-----------|--------|",
        "| GraphSAGE encoder | ✅ Trained — AUC-PR 0.9299 |",
        "| BiLSTM encoder | ✅ Trained — AUC-PR 0.9380 |",
        "| DistilBERT encoder | ✅ Trained — AUC-PR 0.84 |",
        f"| Late-fusion MLP | {'✅ Trained — AUC-PR ' + str(fu['auc_pr']) if fu['auc_pr'] > 0 else '🔲 Pending training run'} |",
        "| SHAP force plots | 🔲 Pending |",
        "",
        "---",
        "_Generated by [Data2Deploy](https://github.com/jpmartin22/Multimodal_Anti_Money_Laundering)_",
    ]

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    data = load_metrics()
    chart1 = plot_auc_pr(data)
    chart2 = plot_multi_metric(data)
    report = build_report(data, chart1, chart2)

    out = ROOT / "reports" / "cml_report.md"
    out.write_text(report)
    print(f"CML report written → {out}")
    print(f"Charts: {chart1}, {chart2}")


if __name__ == "__main__":
    main()
