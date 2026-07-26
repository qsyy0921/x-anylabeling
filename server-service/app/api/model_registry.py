from __future__ import annotations

import asyncio
import os
import re
import secrets
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Header, HTTPException

from app.schemas.response import SuccessResponse

router = APIRouter()

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
WEIGHT_SUFFIXES = {
    ".bin",
    ".data",
    ".engine",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_INSTALL_LOCK = threading.Lock()


def _root() -> Path:
    return Path(
        os.getenv("AUTOLABEL_ROOT", "/data/mfl/autolabel")
    ).expanduser().resolve()


def _catalog_path() -> Path:
    return Path(
        os.getenv(
            "AUTOLABEL_MODEL_REGISTRY",
            str(_root() / "server/configs/langgao-model-registry.yaml"),
        )
    ).expanduser().resolve()


def _models_root() -> Path:
    root = Path(
        os.getenv("AUTOLABEL_MODELS_ROOT", str(_root() / "models"))
    ).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _models_config_path() -> Path:
    return Path(
        os.getenv(
            "XANYLABELING_MODELS_CONFIG",
            str(_root() / "server/configs/langgao-models.yaml"),
        )
    ).expanduser().resolve()


def _require_admin_token(token: str) -> None:
    expected = os.getenv("AUTOLABEL_MODEL_UPLOAD_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=503, detail="Model registry administration is disabled"
        )
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=403, detail="Model registry administration is not allowed"
        )


def _safe_model_id(value: str) -> str:
    if not MODEL_ID_PATTERN.fullmatch(value or ""):
        raise HTTPException(status_code=400, detail="Invalid model ID")
    return value


def _resolve_target(model_id: str, target_dir: str | None = None) -> Path:
    directory = _safe_model_id(target_dir or model_id)
    root = _models_root()
    target = (root / directory).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Model target escapes the model root"
        ) from exc
    return target


def _load_catalog() -> list[dict[str, Any]]:
    path = _catalog_path()
    if not path.is_file():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get("models", [])
    if not isinstance(entries, list):
        raise HTTPException(status_code=500, detail="Invalid model registry")

    validated = []
    seen = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise HTTPException(status_code=500, detail="Invalid model entry")
        entry = dict(raw_entry)
        model_id = _safe_model_id(str(entry.get("model_id", "")))
        if model_id in seen:
            raise HTTPException(
                status_code=500, detail=f"Duplicate registry model: {model_id}"
            )
        seen.add(model_id)
        entry["model_id"] = model_id
        entry["target_dir"] = _safe_model_id(
            str(entry.get("target_dir") or model_id)
        )
        validated.append(entry)
    return validated


def _find_catalog_entry(model_id: str) -> dict[str, Any]:
    model_id = _safe_model_id(model_id)
    for entry in _load_catalog():
        if entry["model_id"] == model_id:
            return entry
    raise HTTPException(status_code=404, detail="Registry model not found")


def _read_enabled_models() -> list[str]:
    path = _models_config_path()
    if not path.is_file():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    enabled = payload.get("enabled_models", [])
    if not isinstance(enabled, list):
        raise HTTPException(status_code=500, detail="Invalid enabled-model config")
    return [str(item) for item in enabled]


def _write_enabled_models(enabled: list[str]) -> None:
    path = _models_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(
                {"enabled_models": enabled},
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _has_weights(target: Path) -> bool:
    return target.is_dir() and any(
        item.is_file() and item.suffix.lower() in WEIGHT_SUFFIXES
        for item in target.rglob("*")
    )


def _entry_status(entry: dict[str, Any]) -> dict[str, Any]:
    from app.main import loader

    model_id = entry["model_id"]
    target = _resolve_target(model_id, entry.get("target_dir"))
    enabled = model_id in _read_enabled_models()
    registered = bool(loader and model_id in loader.model_registry)
    loaded = bool(loader and model_id in loader.models)
    config_path = (
        _root() / "server/configs/auto_labeling" / f"{model_id}.yaml"
    )
    source = entry.get("source") or {}
    return {
        "model_id": model_id,
        "display_name": entry.get("display_name", model_id),
        "description": entry.get("description", ""),
        "source": {
            "provider": source.get("provider", ""),
            "model_id": source.get("model_id", ""),
            "revision": source.get("revision", "master"),
        },
        "installed": _has_weights(target),
        "enabled": enabled,
        "registered": registered,
        "loaded": loaded,
        "config_available": config_path.is_file(),
        "restart_required": enabled and not loaded,
    }


def _install_model(entry: dict[str, Any]) -> dict[str, Any]:
    model_id = entry["model_id"]
    target = _resolve_target(model_id, entry.get("target_dir"))
    if _has_weights(target):
        return {"model_id": model_id, "status": "already_installed"}

    source = entry.get("source") or {}
    provider = str(source.get("provider", "")).lower()
    source_id = str(source.get("model_id", ""))
    revision = str(source.get("revision", "master"))
    if provider != "modelscope":
        raise HTTPException(
            status_code=400,
            detail="Only curated ModelScope registry entries can be installed",
        )
    if (
        not SOURCE_ID_PATTERN.fullmatch(source_id)
        or ".." in Path(source_id).parts
    ):
        raise HTTPException(status_code=500, detail="Invalid registry source ID")

    temporary_root = _root() / "model_downloads"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root / f".{model_id}-{uuid.uuid4().hex}"
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            model_id=source_id,
            revision=revision,
            local_dir=str(temporary),
        )
        if not _has_weights(temporary):
            raise HTTPException(
                status_code=422,
                detail="Downloaded registry package contains no model weights",
            )
        if target.exists():
            raise HTTPException(
                status_code=409, detail="Model target already exists"
            )
        temporary.replace(target)
        target.chmod(0o700)
        return {"model_id": model_id, "status": "installed"}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


@router.get("/v1/model-registry")
async def list_model_registry():
    """List curated server-side models and their installation state."""
    models = [_entry_status(entry) for entry in _load_catalog()]
    return SuccessResponse(data={"models": models})


@router.post("/v1/model-registry/{model_id}/install")
async def install_registry_model(
    model_id: str,
    admin_token: str = Header(default="", alias="X-Model-Upload-Token"),
):
    """Install a curated model from ModelScope into server model storage."""
    _require_admin_token(admin_token)
    entry = _find_catalog_entry(model_id)
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="Another model installation is running"
        )
    try:
        result = await asyncio.to_thread(_install_model, entry)
    finally:
        _INSTALL_LOCK.release()
    return SuccessResponse(data={**result, **_entry_status(entry)})


@router.post("/v1/model-registry/{model_id}/enable")
async def enable_registry_model(
    model_id: str,
    admin_token: str = Header(default="", alias="X-Model-Upload-Token"),
):
    """Enable an installed registered model for the next service restart."""
    _require_admin_token(admin_token)
    entry = _find_catalog_entry(model_id)
    status = _entry_status(entry)
    if not status["installed"]:
        raise HTTPException(status_code=409, detail="Model is not installed")
    if not status["registered"] or not status["config_available"]:
        raise HTTPException(
            status_code=409,
            detail="Model implementation or server configuration is unavailable",
        )

    enabled = _read_enabled_models()
    if model_id not in enabled:
        enabled.append(model_id)
        _write_enabled_models(enabled)
    status = _entry_status(entry)
    status["restart_required"] = not status["loaded"]
    return SuccessResponse(data=status)
