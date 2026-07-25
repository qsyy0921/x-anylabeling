import zipfile

import pytest
from fastapi import HTTPException

from app.api.storage import _resolve_under, _validate_model_archive


def test_resolve_under_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException) as error:
        _resolve_under(tmp_path, "../outside")
    assert error.value.status_code == 400


def test_model_archive_requires_config_and_weights(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "missing model files")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(HTTPException) as error:
            _validate_model_archive(archive)
    assert error.value.status_code == 400


def test_model_archive_accepts_safe_weight_package(tmp_path):
    archive_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("config.yaml", "type: test\n")
        archive.writestr("weights/model.onnx", b"test")

    with zipfile.ZipFile(archive_path) as archive:
        members = _validate_model_archive(archive)
    assert {member.filename for member in members} == {
        "config.yaml",
        "weights/model.onnx",
    }
