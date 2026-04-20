"""Tests for the default Model scaffold."""

from __future__ import annotations

from pathlib import Path

import pytest

from multimodal_anti_money_laundering.models.base import BaseModel
from multimodal_anti_money_laundering.models.model import Model


class TestModel:
    def test_is_base_model(self) -> None:
        assert issubclass(Model, BaseModel)

    def test_default_config_is_empty_dict(self) -> None:
        assert Model().config == {}

    def test_custom_config_is_stored(self) -> None:
        cfg = {"lr": 0.01, "epochs": 5}
        assert Model(cfg).config == cfg

    def test_fit_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Model().fit(None, None)

    def test_predict_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            Model().predict(None)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "model.joblib"
        original = Model({"lr": 0.05})
        original.save(path)

        loaded = Model.load(path)
        assert isinstance(loaded, Model)
        assert loaded.config == original.config

    def test_load_rejects_wrong_type(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"just": "a dict"}, path)

        with pytest.raises(TypeError):
            Model.load(path)
