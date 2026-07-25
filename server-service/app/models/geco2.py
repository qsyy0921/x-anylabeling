import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger

from app.core.registry import register_model
from app.models import BaseModel
from app.schemas.shape import Shape


@register_model("geco2")
class GECO2(BaseModel):
    """GECO2 few/zero-shot counting model (PyTorch inference only)."""

    @staticmethod
    def _resolve_path(path_value: str | None) -> str | None:
        if not path_value:
            return None

        path = Path(path_value)
        if path.is_absolute():
            return str(path)

        project_root = Path(__file__).resolve().parents[2]
        return str((project_root / path).resolve())

    def load(self):
        sam2_parent_dir = os.path.dirname(os.path.abspath(__file__))
        if sam2_parent_dir not in sys.path:
            sys.path.insert(0, sam2_parent_dir)

        from .geco2_impl.counter import build_model

        model_path = self._resolve_path(self.params.get("model_path"))
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GECO2 checkpoint not found: {model_path!r}"
            )

        sam2_ckpt = self._resolve_path(self.params.get("sam2_checkpoint_path"))
        if sam2_ckpt and not os.path.exists(sam2_ckpt):
            logger.warning(
                f"SAM2 checkpoint not found at {sam2_ckpt}, "
                "mask decoder weights will be loaded from the GECO2 checkpoint only"
            )
            sam2_ckpt = None

        device_cfg = self.params.get("device", "cuda:0")
        if "cuda" in device_cfg and torch.cuda.is_available():
            self.device = torch.device(
                device_cfg if ":" in device_cfg else "cuda:0"
            )
        else:
            if "cuda" in device_cfg:
                logger.warning(
                    "CUDA requested but not available, falling back to CPU"
                )
            self.device = torch.device("cpu")

        image_size = int(self.params.get("image_size", 1024))
        reduction = int(self.params.get("reduction", 16))
        emb_dim = int(self.params.get("emb_dim", 256))
        self.image_size = image_size

        logger.info(
            f"Loading GECO2 model weights from {model_path} on {self.device}"
        )
        model = build_model(
            image_size=image_size,
            reduction=reduction,
            emb_dim=emb_dim,
            zero_shot=True,
        )

        if sam2_ckpt:
            sam2_checkpoint = torch.load(
                sam2_ckpt, map_location="cpu", weights_only=True
            )
            if (
                isinstance(sam2_checkpoint, dict)
                and "model" in sam2_checkpoint
            ):
                sam2_checkpoint = sam2_checkpoint["model"]
            backbone_sd = {
                k.replace("image_encoder.", ""): v
                for k, v in sam2_checkpoint.items()
            }
            model.backbone.load_state_dict(backbone_sd, strict=False)
            model.sam_mask.load_sam2_weights(sam2_checkpoint)

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = (
            ckpt["model"]
            if isinstance(ckpt, dict) and "model" in ckpt
            else ckpt
        )
        state_dict = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

        logger.info("GECO2 model loaded successfully")

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        from torchvision import transforms as T
        from torchvision.ops import nms

        from .geco2_impl.data_utils import resize_and_pad

        marks = params.get("marks") or []
        boxes: List[List[float]] = []
        for mark in marks:
            if mark.get("type") != "rectangle":
                continue
            data = mark.get("data") or []
            if len(data) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in data)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])

        if not boxes:
            logger.warning(
                "GECO2 predict called without valid rectangle prompts"
            )
            return {"shapes": [], "description": "", "replace": False}

        threshold = float(
            params.get(
                "conf_threshold", self.params.get("conf_threshold", 0.33)
            )
        )
        threshold = min(max(threshold, 0.05), 0.95)

        self.model.return_masks = False

        image_rgb = image[:, :, ::-1].copy()
        image_tensor = (
            torch.from_numpy(image_rgb)
            .to(self.device)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        image_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image_tensor)

        bboxes_tensor = torch.tensor(
            boxes, dtype=torch.float32, device=self.device
        )

        img, bboxes_rp, scale = resize_and_pad(
            image_tensor, bboxes_tensor, size=float(self.image_size)
        )
        img = img.unsqueeze(0).to(self.device)
        bboxes_rp = bboxes_rp.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, _, _, _, _ = self.model(img, bboxes_rp)

        inv_thr = 1.0 / threshold
        out = outputs[0]
        box_v = out["box_v"]
        if box_v.numel() == 0:
            return {"shapes": [], "description": "count: 0", "replace": False}

        mask = box_v > (box_v.max() / inv_thr)
        pred_boxes = out["pred_boxes"][mask]
        filtered_scores = box_v[mask]
        if pred_boxes.numel() == 0:
            return {"shapes": [], "description": "count: 0", "replace": False}

        keep = nms(pred_boxes, filtered_scores.flatten(), 0.5)
        pred_boxes = torch.clamp(pred_boxes[keep], 0.0, 1.0)
        pred_boxes = (pred_boxes.cpu() / scale * img.shape[-1]).tolist()

        shapes = []
        image_height, image_width = image.shape[:2]
        for box in pred_boxes:
            x1, y1, x2, y2 = box
            x1 = min(max(float(x1), 0.0), image_width - 1)
            y1 = min(max(float(y1), 0.0), image_height - 1)
            x2 = min(max(float(x2), 0.0), image_width - 1)
            y2 = min(max(float(y2), 0.0), image_height - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            shapes.append(
                Shape(
                    label="AUTOLABEL_OBJECT",
                    shape_type="rectangle",
                    points=[
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                )
            )

        return {"shapes": shapes, "description": ""}

    def unload(self):
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
