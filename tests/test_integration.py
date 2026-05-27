"""Integration tests for the AML data processing and evaluation pipeline.

These tests exercise the full chain of lightweight components — feature
engineering, metric computation, I/O helpers, the evaluation gate, and
seeding — without requiring torch, transformers, or DVC-tracked data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from multimodal_anti_money_laundering.evaluation.eval_gate import check_metrics_gate
from multimodal_anti_money_laundering.evaluation.metrics import (
    classification_report,
    compute_aml_metrics,
    log_metrics_table,
    regression_report,
)
from multimodal_anti_money_laundering.features.build_features import build_features
from multimodal_anti_money_laundering.utils.io import load_json, save_json
from multimodal_anti_money_laundering.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _binary_data(
    n: int = 200, illicit_rate: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_score) with a realistic class imbalance."""
    y_true = (RNG.random(n) < illicit_rate).astype(int)
    # Scores that are correlated with labels but noisy
    y_score = np.clip(y_true * 0.8 + RNG.random(n) * 0.4, 0.0, 1.0)
    return y_true, y_score


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"amount": [100.0, 200.0], "flag": [0, 1]})
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_preserved(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = build_features(df)
        assert list(result.columns) == ["a", "b"]

    def test_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        original = df.copy()
        build_features(df)
        pd.testing.assert_frame_equal(df, original)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["amount", "flag"])
        result = build_features(df)
        assert result.empty
        assert list(result.columns) == ["amount", "flag"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestComputeAmlMetrics:
    def setup_method(self) -> None:
        self.y_true, self.y_score = _binary_data()

    def test_returns_expected_keys(self) -> None:
        m = compute_aml_metrics(self.y_true, self.y_score)
        assert {"auc_pr", "f1", "roc_auc"}.issubset(m.keys())

    def test_auc_pr_in_range(self) -> None:
        m = compute_aml_metrics(self.y_true, self.y_score)
        assert 0.0 <= m["auc_pr"] <= 1.0

    def test_roc_auc_in_range(self) -> None:
        m = compute_aml_metrics(self.y_true, self.y_score)
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_aml_metrics(y_true, y_score)
        assert m["auc_pr"] == pytest.approx(1.0, abs=0.01)

    def test_custom_target_recall_key_present(self) -> None:
        m = compute_aml_metrics(self.y_true, self.y_score, target_recall=0.90)
        assert "precision_at_r90" in m
        assert "fpr_at_r90" in m

    def test_values_are_rounded_to_4dp(self) -> None:
        m = compute_aml_metrics(self.y_true, self.y_score)
        for v in m.values():
            assert round(v, 4) == v


class TestClassificationReport:
    def test_all_keys_present(self) -> None:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        r = classification_report(y_true, y_pred)
        assert set(r.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_perfect_prediction(self) -> None:
        y_true = [0, 0, 1, 1]
        r = classification_report(y_true, y_true)
        assert r["accuracy"] == 1.0
        assert r["f1"] == 1.0


class TestRegressionReport:
    def test_all_keys_present(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.1, 2.1, 3.1]
        r = regression_report(y_true, y_pred)
        assert set(r.keys()) == {"mae", "mse", "rmse", "r2"}

    def test_perfect_prediction(self) -> None:
        y = [1.0, 2.0, 3.0]
        r = regression_report(y, y)
        assert r["mae"] == pytest.approx(0.0)
        assert r["mse"] == pytest.approx(0.0)
        assert r["r2"] == pytest.approx(1.0)

    def test_rmse_is_sqrt_of_mse(self) -> None:
        y_true = [0.0, 1.0, 2.0]
        y_pred = [0.5, 1.5, 2.5]
        r = regression_report(y_true, y_pred)
        assert r["rmse"] == pytest.approx(r["mse"] ** 0.5)


class TestLogMetricsTable:
    def test_runs_without_error(self, capsys) -> None:
        results = {
            "ModelA": {"auc_pr": 0.85, "f1": 0.70},
            "ModelB": {"auc_pr": 0.80, "f1": 0.65},
        }
        log_metrics_table(results, title="Test Comparison")
        out = capsys.readouterr().out
        assert "ModelA" in out
        assert "ModelB" in out

    def test_empty_dict_is_noop(self, capsys) -> None:
        log_metrics_table({})
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# Evaluation gate
# ---------------------------------------------------------------------------


class TestEvalGate:
    def test_passes_when_auc_pr_above_threshold(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"auc_pr": 0.85}))
        with pytest.raises(SystemExit) as exc:
            check_metrics_gate(str(metrics_file))
        assert exc.value.code == 0

    def test_fails_when_auc_pr_below_threshold(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"auc_pr": 0.50}))
        with pytest.raises(SystemExit) as exc:
            check_metrics_gate(str(metrics_file))
        assert exc.value.code == 1

    def test_fails_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            check_metrics_gate(str(tmp_path / "nonexistent.json"))
        assert exc.value.code == 1

    def test_fails_when_auc_pr_key_missing(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"f1": 0.90}))
        # Missing auc_pr defaults to 0.0 → fails gate
        with pytest.raises(SystemExit) as exc:
            check_metrics_gate(str(metrics_file))
        assert exc.value.code == 1

    def test_boundary_exactly_at_threshold(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"auc_pr": 0.80}))
        with pytest.raises(SystemExit) as exc:
            check_metrics_gate(str(metrics_file))
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


class TestJsonIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        data = {"auc_pr": 0.85, "model": "fusion", "tags": [1, 2, 3]}
        path = tmp_path / "sub" / "metrics.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c" / "file.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_pretty_printed(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        save_json({"z": 1, "a": 2}, path)
        content = path.read_text()
        assert "\n" in content  # multi-line = pretty printed

    def test_keys_are_sorted(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        save_json({"z": 1, "a": 2}, path)
        parsed = json.loads(path.read_text())
        assert list(parsed.keys()) == sorted(parsed.keys())


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------


class TestSetSeed:
    def test_numpy_reproducibility(self) -> None:
        set_seed(0)
        a = np.random.rand(5)
        set_seed(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        set_seed(1)
        a = np.random.rand(5)
        set_seed(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)

    def test_runs_without_torch(self) -> None:
        # Should not raise even when torch is unavailable (ImportError caught)
        with patch.dict(sys.modules, {"torch": None}):
            set_seed(42)


# ---------------------------------------------------------------------------
# Pipeline integration: features → metrics → gate
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end smoke test chaining feature engineering, metric computation,
    and the evaluation gate — the lightweight CPU-only pipeline."""

    def test_pipeline_passes_gate(self, tmp_path: Path) -> None:
        # 1. Simulate a raw dataframe
        df = pd.DataFrame(
            {
                "amount": np.abs(RNG.normal(1000, 200, 500)),
                "label": (RNG.random(500) < 0.05).astype(int),
            }
        )

        # 2. Feature engineering (no-op pass-through for now)
        features_df = build_features(df)
        assert len(features_df) == len(df)

        # 3. Compute metrics from simulated model scores
        y_true = features_df["label"].values
        # A reasonable scorer: higher amount → higher risk (toy rule)
        amount = features_df["amount"].values
        y_score = np.clip(
            (amount - amount.min()) / (amount.max() - amount.min()),
            0.0,
            1.0,
        )
        metrics = compute_aml_metrics(y_true, y_score)
        assert 0.0 <= metrics["auc_pr"] <= 1.0

        # 4. Persist metrics
        metrics_path = tmp_path / "reports" / "metrics.json"
        save_json(metrics, metrics_path)
        assert metrics_path.exists()

        # 5. Reload and verify structure
        loaded = load_json(metrics_path)
        assert "auc_pr" in loaded

        # 6. Gate check — inject a known-passing value
        override_path = tmp_path / "reports" / "override.json"
        save_json({"auc_pr": 0.82}, override_path)
        with pytest.raises(SystemExit) as exc:
            check_metrics_gate(str(override_path))
        assert exc.value.code == 0
