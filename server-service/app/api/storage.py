from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import (
    APIRouter,
    Body,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.schemas.response import SuccessResponse

router = APIRouter()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANNOTATION_SUFFIXES = {".json", ".txt", ".yaml", ".yml"}
MODEL_SUFFIXES = {
    ".onnx",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".engine",
    ".data",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".gz",
}
MAX_MODEL_UPLOAD_BYTES = int(
    os.getenv("AUTOLABEL_MAX_MODEL_UPLOAD_BYTES", str(20 * 1024**3))
)
MAX_MODEL_UNPACKED_BYTES = int(
    os.getenv("AUTOLABEL_MAX_MODEL_UNPACKED_BYTES", str(50 * 1024**3))
)
MAX_MODEL_MEMBERS = int(os.getenv("AUTOLABEL_MAX_MODEL_MEMBERS", "2000"))
_ANNOTATION_LOCKS: dict[str, threading.Lock] = {}
_ANNOTATION_LOCKS_GUARD = threading.Lock()


def _root_from_env(name: str, default: str) -> Path:
    root = Path(os.getenv(name, default)).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_under(root: Path, relative_path: str, must_exist: bool = True) -> Path:
    raw_path = str(relative_path or ".").replace("\\", "/")
    if raw_path.startswith("server://"):
        raw_path = raw_path[len("server://") :]
    candidate_input = Path(raw_path)
    if candidate_input.is_absolute():
        try:
            candidate_input = candidate_input.resolve().relative_to(root)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Path escapes storage root"
            ) from exc
    relative = Path(candidate_input)
    if ".." in relative.parts:
        raise HTTPException(status_code=400, detail="Invalid relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes storage root") from exc
    if must_exist and not candidate.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    return candidate


def _relative_text(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    return "" if value == "." else value


def _file_revision(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_path_for_image(root: Path, image_path: str) -> tuple[Path, Path]:
    image = _resolve_under(root, image_path)
    if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="Path is not an image")
    return image, image.with_suffix(".json")


def _annotation_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _ANNOTATION_LOCKS_GUARD:
        return _ANNOTATION_LOCKS.setdefault(key, threading.Lock())


def _validated_annotation(annotation: dict, image: Path) -> dict:
    if not isinstance(annotation, dict):
        raise HTTPException(status_code=422, detail="Annotation must be an object")
    if not isinstance(annotation.get("shapes", []), list):
        raise HTTPException(status_code=422, detail="Annotation shapes must be a list")
    normalized = dict(annotation)
    normalized["imagePath"] = image.name
    normalized.setdefault("imageData", None)
    normalized.setdefault("flags", {})
    normalized.setdefault("shapes", [])
    return normalized


def _atomic_write_annotation(
    annotation_path: Path,
    annotation: dict,
    expected_revision: str | None,
) -> str:
    lock = _annotation_lock(annotation_path)
    with lock:
        current_revision = _file_revision(annotation_path)
        revision_conflict = (
            expected_revision == "__missing__" and current_revision is not None
        ) or (
            expected_revision not in {None, "*", "__missing__"}
            and expected_revision != current_revision
        )
        if revision_conflict:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Annotation was modified by another client",
                    "current_revision": current_revision,
                },
            )
        payload = json.dumps(
            annotation, ensure_ascii=False, indent=2
        ).encode("utf-8")
        temp_path = annotation_path.with_name(
            f".{annotation_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temp_path.write_bytes(payload)
            os.replace(temp_path, annotation_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return hashlib.sha256(payload).hexdigest()


@router.get("/v1/data/directories")
async def list_data_directories(
    max_depth: int = Query(default=4, ge=1, le=8),
):
    """List server dataset directories without exposing absolute paths."""
    root = _root_from_env("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
    directories = []
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".") and depth < max_depth
        )
        image_count = sum(
            Path(name).suffix.lower() in IMAGE_SUFFIXES for name in filenames
        )
        if depth > 0 and (image_count or depth < max_depth):
            directories.append(
                {
                    "path": _relative_text(current_path, root),
                    "absolute_path": str(current_path),
                    "name": current_path.name,
                    "image_count": image_count,
                }
            )
    return SuccessResponse(data={"directories": directories})


@router.get("/v1/data/files")
async def list_data_files(
    path: str = Query(default=""),
    recursive: bool = Query(default=True),
):
    """List downloadable image and annotation files in a server directory."""
    root = _root_from_env("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
    directory = _resolve_under(root, path)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    files = []
    for item in sorted(iterator):
        if not item.is_file():
            continue
        if item.suffix.lower() not in IMAGE_SUFFIXES | ANNOTATION_SUFFIXES:
            continue
        file_data = {
            "path": _relative_text(item, root),
            "size": item.stat().st_size,
        }
        if item.suffix.lower() in IMAGE_SUFFIXES:
            annotation_path = item.with_suffix(".json")
            file_data["annotation_exists"] = annotation_path.is_file()
            file_data["annotation_revision"] = _file_revision(annotation_path)
        files.append(file_data)
        if len(files) >= 10000:
            raise HTTPException(
                status_code=413,
                detail="Dataset contains more than 10000 supported files",
            )
    return SuccessResponse(data={"base_path": path, "files": files})


@router.get("/v1/data/file")
async def download_data_file(path: str = Query(...)):
    """Download one server dataset file."""
    root = _root_from_env("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
    file_path = _resolve_under(root, path)
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    if file_path.suffix.lower() not in IMAGE_SUFFIXES | ANNOTATION_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported dataset file")
    return FileResponse(file_path, filename=file_path.name)


@router.get("/v1/data/annotation")
async def get_data_annotation(image_path: str = Query(...)):
    """Read the annotation paired with one server-side image."""
    root = _root_from_env("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
    image, annotation_path = _annotation_path_for_image(root, image_path)
    if not annotation_path.is_file():
        return SuccessResponse(
            data={
                "image_path": _relative_text(image, root),
                "exists": False,
                "revision": None,
                "annotation": None,
            }
        )
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail="Server annotation is not valid JSON"
        ) from exc
    return SuccessResponse(
        data={
            "image_path": _relative_text(image, root),
            "exists": True,
            "revision": _file_revision(annotation_path),
            "annotation": annotation,
        }
    )


@router.put("/v1/data/annotation")
async def put_data_annotation(
    image_path: str = Query(...),
    annotation: dict = Body(...),
    expected_revision: str | None = Header(
        default=None, alias="If-Match"
    ),
):
    """Atomically save one annotation beside its server-side image."""
    root = _root_from_env("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
    image, annotation_path = _annotation_path_for_image(root, image_path)
    normalized = _validated_annotation(annotation, image)
    revision = _atomic_write_annotation(
        annotation_path, normalized, expected_revision
    )
    return SuccessResponse(
        data={
            "image_path": _relative_text(image, root),
            "annotation_path": _relative_text(annotation_path, root),
            "revision": revision,
        }
    )


@router.delete("/v1/data/annotation")
async def delete_data_annotation(
    image_path: str = Query(...),
    expected_revision: str | None = Header(
        default=None, alias="If-Match"
    ),
):
    """Delete one server annotation with optimistic concurrency control."""
    root = _root_from_env("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
    image, annotation_path = _annotation_path_for_image(root, image_path)
    with _annotation_lock(annotation_path):
        current_revision = _file_revision(annotation_path)
        revision_conflict = (
            expected_revision == "__missing__" and current_revision is not None
        ) or (
            expected_revision not in {None, "*", "__missing__"}
            and expected_revision != current_revision
        )
        if revision_conflict:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Annotation was modified by another client",
                    "current_revision": current_revision,
                },
            )
        annotation_path.unlink(missing_ok=True)
    return SuccessResponse(
        data={
            "image_path": _relative_text(image, root),
            "deleted": current_revision is not None,
        }
    )


def _validate_model_archive(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members or len(members) > MAX_MODEL_MEMBERS:
        raise HTTPException(status_code=400, detail="Invalid archive member count")

    total_size = 0
    has_weight = False
    has_config = False
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise HTTPException(status_code=400, detail="Unsafe archive path")
        if member.is_dir():
            continue
        # Unix symlinks are not accepted.
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise HTTPException(status_code=400, detail="Symlinks are not allowed")
        suffix = Path(path.name).suffix.lower()
        if suffix not in MODEL_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported model package file: {path.name}",
            )
        total_size += member.file_size
        has_weight = has_weight or suffix in {
            ".onnx",
            ".pt",
            ".pth",
            ".bin",
            ".safetensors",
            ".engine",
            ".data",
        }
        has_config = has_config or suffix in {".yaml", ".yml", ".json"}

    if total_size > MAX_MODEL_UNPACKED_BYTES:
        raise HTTPException(status_code=413, detail="Unpacked model is too large")
    if not has_weight or not has_config:
        raise HTTPException(
            status_code=400,
            detail="Model package must contain weights and a YAML/JSON config",
        )
    return members


def _safe_package_name(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return stem[:80] or "model"


@router.post("/v1/models/upload")
async def upload_model_package(
    file: UploadFile = File(...),
    upload_token: str = Header(default="", alias="X-Model-Upload-Token"),
):
    """Upload a model package to server staging without executing it."""
    expected_token = os.getenv("AUTOLABEL_MODEL_UPLOAD_TOKEN", "")
    if not expected_token:
        raise HTTPException(status_code=503, detail="Model upload is disabled")
    if not secrets.compare_digest(upload_token, expected_token):
        raise HTTPException(status_code=403, detail="Model upload is not allowed")
    if not file.filename or Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(status_code=415, detail="Only ZIP packages are accepted")

    upload_root = _root_from_env(
        "AUTOLABEL_MODEL_UPLOAD_ROOT",
        "/data/mfl/autolabel/model_uploads/incoming",
    )
    package_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{_safe_package_name(file.filename)}-{uuid.uuid4().hex[:8]}"
    )
    target = _resolve_under(upload_root, package_id, must_exist=False)
    temp_file = upload_root / f".{package_id}.upload"
    received = 0

    try:
        with temp_file.open("wb") as output:
            while chunk := await file.read(8 * 1024**2):
                received += len(chunk)
                if received > MAX_MODEL_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Model package is too large"
                    )
                output.write(chunk)

        with zipfile.ZipFile(temp_file) as archive:
            members = _validate_model_archive(archive)
            target.mkdir(mode=0o700)
            for member in members:
                if member.is_dir():
                    continue
                member_path = target.joinpath(*PurePosixPath(member.filename).parts)
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, member_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)

        metadata = {
            "package_id": package_id,
            "original_filename": file.filename,
            "uploaded_bytes": received,
            "status": "staged",
            "note": "Administrative review is required before activation.",
        }
        (target / "upload.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return SuccessResponse(data=metadata)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid ZIP package") from exc
    finally:
        await file.close()
        temp_file.unlink(missing_ok=True)
