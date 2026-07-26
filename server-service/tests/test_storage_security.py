import zipfile

import pytest
from fastapi import HTTPException

from app.api.storage import (
    _annotation_path_for_image,
    _atomic_write_annotation,
    _resolve_under,
    _validate_model_archive,
)


def test_resolve_under_rejects_traversal(tmp_path):
    with pytest.raises(HTTPException) as error:
        _resolve_under(tmp_path, "../outside")
    assert error.value.status_code == 400


def test_resolve_under_accepts_absolute_path_inside_root(tmp_path):
    image = tmp_path / "dataset" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")

    assert _resolve_under(tmp_path, str(image)) == image


def test_annotation_write_is_atomic_and_detects_stale_revision(tmp_path):
    image = tmp_path / "dataset" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    _, annotation_path = _annotation_path_for_image(
        tmp_path, "dataset/image.jpg"
    )
    first_revision = _atomic_write_annotation(
        annotation_path,
        {"imagePath": "image.jpg", "shapes": []},
        "__missing__",
    )

    assert annotation_path.is_file()
    with pytest.raises(HTTPException) as error:
        _atomic_write_annotation(
            annotation_path,
            {"imagePath": "image.jpg", "shapes": [{"label": "new"}]},
            "stale-revision",
        )
    assert error.value.status_code == 409

    second_revision = _atomic_write_annotation(
        annotation_path,
        {"imagePath": "image.jpg", "shapes": [{"label": "new"}]},
        first_revision,
    )
    assert second_revision != first_revision


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
