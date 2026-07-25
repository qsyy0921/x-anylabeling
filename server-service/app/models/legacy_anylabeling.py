from __future__ import annotations

import gc
import hashlib
import importlib
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml
from loguru import logger
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QImage

import qimage2ndarray
import anylabeling.config as app_config
from app.core.registry import register_model
from app.models import BaseModel


ROOT = Path(os.getenv("AUTOLABEL_ROOT", "/data/mfl/autolabel")).resolve()

MODEL_SPECS = {
    "edge_sam": {
        "module": "anylabeling.services.auto_labeling.edge_sam",
        "class": "EdgeSAM",
        "config": ROOT / "models/edge_sam/config.local.yaml",
        "runtime": "onnx",
        "mode": "visual",
    },
    "geco_sam_hq_vit_h": {
        "module": "anylabeling.services.auto_labeling.geco",
        "class": "GeCo",
        "config": ROOT / "models/geco_sam_hq_vit_h/config.local.yaml",
        "runtime": "onnx",
        "mode": "visual",
    },
    "groundingdino_swint_sam2_large": {
        "module": "anylabeling.services.auto_labeling.grounding_sam2",
        "class": "GroundingSAM2",
        "config": (
            ROOT
            / "models/groundingdino_swint_sam2_large/config.local.yaml"
        ),
        "runtime": "onnx",
        "mode": "grounding",
    },
    "sam2_hiera_tiny": {
        "module": "anylabeling.services.auto_labeling.segment_anything_2",
        "class": "SegmentAnything2",
        "config": ROOT / "models/sam2_hiera_tiny/config.local.yaml",
        "runtime": "onnx",
        "mode": "visual",
    },
    "sam2.1_hiera_large_20260221": {
        "module": "anylabeling.services.auto_labeling.segment_anything_2",
        "class": "SegmentAnything2",
        "config": (
            ROOT / "models/sam2.1_hiera_large_20260221/config.yaml"
        ),
        "runtime": "onnx",
        "mode": "visual",
    },
    "sam3_vit_h": {
        "module": "anylabeling.services.auto_labeling.segment_anything_3",
        "class": "SegmentAnything3",
        "config": ROOT / "models/sam3_vit_h/config.local.yaml",
        "runtime": "onnx",
        "mode": "sam3_text",
    },
    "sam3.1_multiplex_official": {
        "module": "anylabeling.services.auto_labeling.sam31_pytorch",
        "class": "Sam31PyTorch",
        "config": (
            ROOT / "models/sam3.1_multiplex_official/config.yaml"
        ),
        "runtime": "pytorch",
        "mode": "visual",
    },
    "sam_hq_vit_l": {
        "module": "anylabeling.services.auto_labeling.sam_hq",
        "class": "SAM_HQ",
        "config": ROOT / "models/sam_hq_vit_l/config.local.yaml",
        "runtime": "onnx",
        "mode": "visual",
    },
    "yolov8s_sam2_hiera_base": {
        "module": "anylabeling.services.auto_labeling.yolov8_sam2",
        "class": "YOLOv8SegmentAnything2",
        "config": (
            ROOT / "models/yolov8s_sam2_hiera_base/config.local.yaml"
        ),
        "runtime": "onnx",
        "mode": "automatic",
    },
}


def _find_sessions(
    value: object, seen: set[int], depth: int = 0
) -> list[object]:
    if value is None or depth > 5 or id(value) in seen:
        return []
    seen.add(id(value))
    if callable(getattr(value, "get_providers", None)):
        return [value]
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple, set)):
        children = value
    elif hasattr(value, "__dict__"):
        children = vars(value).values()
    else:
        return []

    sessions = []
    for child in children:
        sessions.extend(_find_sessions(child, seen, depth + 1))
    return sessions


def _assert_gpu_only(model: object, runtime: str, model_id: str) -> None:
    if runtime == "onnx":
        sessions = _find_sessions(model, set())
        if not sessions:
            raise RuntimeError(
                f"GPU-only check failed for {model_id}: "
                "no ONNX Runtime sessions were found"
            )
        providers = [session.get_providers() for session in sessions]
        if not all("CUDAExecutionProvider" in item for item in providers):
            raise RuntimeError(
                f"GPU-only check failed for {model_id}: {providers}"
            )
        logger.info(
            "GPU-only check passed for {}: {} CUDA session(s)",
            model_id,
            len(sessions),
        )
        return

    import torch

    torch_model = getattr(getattr(model, "model", None), "model", None)
    if torch_model is None:
        raise RuntimeError(
            f"GPU-only check failed for {model_id}: "
            "PyTorch model was not found"
        )
    devices = sorted({str(item.device) for item in torch_model.parameters()})
    if not devices or any(not item.startswith("cuda") for item in devices):
        raise RuntimeError(
            f"GPU-only check failed for {model_id}: {devices}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU-only check failed for {model_id}: CUDA is unavailable"
        )
    logger.info(
        "GPU-only check passed for {}: devices={}", model_id, devices
    )


def _to_qimage(image_bgr: np.ndarray) -> QImage:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    return qimage2ndarray.array2qimage(image_rgb)


def _safe_score(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    return min(1.0, max(0.0, score))


def _shape_to_dict(shape: object) -> Dict[str, Any] | None:
    if shape is None:
        return None
    points = [
        [float(point.x()), float(point.y())]
        for point in getattr(shape, "points", [])
    ]
    if not points:
        return None

    direction = float(getattr(shape, "direction", 0.0) or 0.0)
    direction %= 2 * math.pi
    group_id = getattr(shape, "group_id", None)
    if not isinstance(group_id, int) or group_id <= 0:
        group_id = None

    return {
        "label": str(getattr(shape, "label", "AUTOLABEL_OBJECT")),
        "shape_type": str(getattr(shape, "shape_type", "polygon")),
        "points": points,
        "score": _safe_score(getattr(shape, "score", None)),
        "attributes": getattr(shape, "attributes", {}) or {},
        "description": getattr(shape, "description", None),
        "difficult": bool(getattr(shape, "difficult", False)),
        "direction": direction,
        "flags": getattr(shape, "flags", None),
        "group_id": group_id,
        "kie_linking": getattr(shape, "kie_linking", []) or [],
    }


class _LazyModelManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.current_id: str | None = None
        self.current_model: object | None = None
        self.messages: list[str] = []
        self.qt_app = QCoreApplication.instance() or QCoreApplication([])
        app_config.current_config_file = str(ROOT / ".xanylabelingrc")

    def get(self, model_id: str) -> object:
        with self.lock:
            if self.current_id == model_id and self.current_model is not None:
                return self.current_model
            self._unload_locked()

            spec = MODEL_SPECS[model_id]
            config_path = Path(spec["config"])
            model_config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
            model_config["config_file"] = str(config_path)
            model_class = getattr(
                importlib.import_module(spec["module"]), spec["class"]
            )
            logger.info("Lazy loading model [{}]...", model_id)
            model = model_class(model_config, self.messages.append)
            _assert_gpu_only(model, spec["runtime"], model_id)
            self.current_id = model_id
            self.current_model = model
            logger.info("Model [{}] is ready on GPU", model_id)
            return model

    def unload(self) -> None:
        with self.lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        model = self.current_model
        model_id = self.current_id
        self.current_model = None
        self.current_id = None
        if model is not None:
            logger.info("Unloading model [{}]...", model_id)
            if callable(getattr(model, "unload", None)):
                model.unload()
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


_MANAGER = _LazyModelManager()


@register_model(*MODEL_SPECS.keys())
class ExistingAnyLabelingModel(BaseModel):
    """Expose the verified desktop models through the official server API."""

    def load(self) -> None:
        spec = MODEL_SPECS.get(self.model_id)
        if spec is None:
            raise ValueError(f"Unsupported model: {self.model_id}")
        if not Path(spec["config"]).is_file():
            raise FileNotFoundError(spec["config"])
        logger.info("Registered lazy GPU model [{}]", self.model_id)

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        with _MANAGER.lock:
            model = _MANAGER.get(self.model_id)
            spec = MODEL_SPECS[self.model_id]
            marks = params.get("marks", []) or []
            text_prompt = str(params.get("text_prompt", "") or "").strip()
            qimage = _to_qimage(image)
            image_key = "remote-" + hashlib.sha256(
                image.tobytes()
            ).hexdigest()

            if callable(getattr(model, "set_auto_labeling_marks", None)):
                model.set_auto_labeling_marks(marks)

            conf = params.get("conf_threshold")
            if conf is not None and float(conf) > 0:
                setter = getattr(model, "set_auto_labeling_conf", None)
                if callable(setter):
                    setter(float(conf))

            epsilon = params.get("epsilon_factor")
            setter = getattr(model, "set_mask_fineness", None)
            if epsilon is not None and callable(setter):
                setter(float(epsilon))

            output_mode = str(params.get("output_mode", "polygon"))
            supported_modes = getattr(
                getattr(model, "Meta", object), "output_modes", {}
            )
            if output_mode in supported_modes:
                model.output_mode = output_mode

            mode = spec["mode"]
            if mode == "grounding":
                result = model.predict_shapes(
                    qimage,
                    image_path=image_key,
                    text_prompt=text_prompt,
                )
            elif mode == "sam3_text":
                result = model.predict_shapes(
                    qimage,
                    filename=image_key,
                    text_prompt=text_prompt,
                )
            else:
                result = model.predict_shapes(qimage, filename=image_key)

            if isinstance(result, list):
                shapes = result
                description = ""
                replace = False
            else:
                shapes = getattr(result, "shapes", []) or []
                description = getattr(result, "description", "") or ""
                replace = bool(getattr(result, "replace", False))

            converted = []
            for shape in shapes:
                item = _shape_to_dict(shape)
                if item is not None:
                    converted.append(item)
            return {
                "shapes": converted,
                "description": description,
                "replace": replace,
            }

    def unload(self) -> None:
        _MANAGER.unload()
