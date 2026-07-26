from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import model_registry


def test_safe_model_id_rejects_path_traversal():
    with pytest.raises(HTTPException) as error:
        model_registry._safe_model_id("../model")
    assert error.value.status_code == 400


def test_resolve_target_stays_under_model_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_MODELS_ROOT", str(tmp_path))
    target = model_registry._resolve_target("sam_test")
    assert target == tmp_path / "sam_test"


def test_enabled_models_are_written_atomically(tmp_path, monkeypatch):
    config_path = tmp_path / "models.yaml"
    monkeypatch.setenv("XANYLABELING_MODELS_CONFIG", str(config_path))

    model_registry._write_enabled_models(["sam_test"])

    assert model_registry._read_enabled_models() == ["sam_test"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_catalog_rejects_duplicate_models(tmp_path, monkeypatch):
    catalog = tmp_path / "registry.yaml"
    catalog.write_text(
        "models:\n"
        "  - {model_id: duplicate}\n"
        "  - {model_id: duplicate}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTOLABEL_MODEL_REGISTRY", str(catalog))

    with pytest.raises(HTTPException) as error:
        model_registry._load_catalog()
    assert error.value.status_code == 500


def test_has_weights_requires_supported_weight_file(tmp_path):
    target = Path(tmp_path) / "model"
    target.mkdir()
    (target / "config.yaml").write_text("type: test\n", encoding="utf-8")
    assert not model_registry._has_weights(target)

    (target / "weights.onnx").write_bytes(b"model")
    assert model_registry._has_weights(target)
