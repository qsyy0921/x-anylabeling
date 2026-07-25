import gc
import os
import sys
import traceback
from contextlib import nullcontext

import numpy as np
from PIL import Image
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .segment_anything import SegmentAnything
from .types import AutoLabelingResult


class OfficialSam31Runtime:
    """Adapter for Meta's official SAM 3.1 image model."""

    def __init__(
        self,
        checkpoint_path,
        source_path,
        device="cuda",
        confidence_threshold=0.5,
    ):
        if source_path and source_path not in sys.path:
            sys.path.insert(0, source_path)

        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not str(device).startswith("cuda"):
            raise RuntimeError(
                f"SAM 3.1 is configured for GPU-only inference, got {device!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1 requires a CUDA-enabled PyTorch runtime")

        self.torch = torch
        self.device = device
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=1008,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def _autocast(self):
        if str(self.device).startswith("cuda"):
            return self.torch.autocast(
                device_type="cuda", dtype=self.torch.bfloat16
            )
        return nullcontext()

    def encode(self, cv_image):
        if isinstance(cv_image, np.ndarray):
            if cv_image.ndim == 2:
                processor_image = Image.fromarray(
                    np.ascontiguousarray(cv_image)
                )
            elif cv_image.ndim == 3 and cv_image.shape[2] >= 3:
                rgb_image = cv_image[:, :, :3][:, :, ::-1]
                processor_image = Image.fromarray(
                    np.ascontiguousarray(rgb_image)
                )
            else:
                raise ValueError(
                    f"Unsupported image shape for SAM 3.1: {cv_image.shape}"
                )
        else:
            processor_image = cv_image

        with self._autocast():
            return self.processor.set_image(processor_image)

    def add_language_prompt(self, embedding, text_prompt="visual"):
        state = {
            "original_height": embedding["original_height"],
            "original_width": embedding["original_width"],
            "backbone_out": dict(embedding["backbone_out"]),
        }
        with self._autocast():
            return self.processor.set_text_prompt(text_prompt, state)

    def predict_masks(self, state, marks, confidence_threshold=0.5):
        self.processor.set_confidence_threshold(
            max(float(confidence_threshold), 0.5)
        )
        height = max(int(state["original_height"]), 1)
        width = max(int(state["original_width"]), 1)

        for mark in marks or []:
            mark_type = mark.get("type")
            data = mark.get("data", [])
            is_positive = bool(mark.get("label", 1))
            if mark_type == "rectangle" and len(data) >= 4:
                x1, x2 = sorted((float(data[0]), float(data[2])))
                y1, y2 = sorted((float(data[1]), float(data[3])))
                box = [
                    ((x1 + x2) / 2.0) / width,
                    ((y1 + y2) / 2.0) / height,
                    max((x2 - x1) / width, 1.0 / width),
                    max((y2 - y1) / height, 1.0 / height),
                ]
            elif mark_type == "point" and len(data) >= 2:
                box = [
                    float(data[0]) / width,
                    float(data[1]) / height,
                    0.01,
                    0.01,
                ]
            else:
                continue

            box = [min(max(value, 0.0), 1.0) for value in box]
            with self._autocast():
                state = self.processor.add_geometric_prompt(
                    box=box,
                    label=is_positive,
                    state=state,
                )

        masks = state.get("masks")
        if masks is None:
            return np.zeros((0, 1, height, width), dtype=np.bool_)
        return masks.detach().to("cpu").numpy()

    def unload(self):
        self.processor = None
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class Sam31PyTorch(SegmentAnything):
    """Expose official SAM 3.1 through X-AnyLabeling visual prompts."""

    class Meta(SegmentAnything.Meta):
        required_config_names = [
            "type",
            "name",
            "display_name",
            "checkpoint_path",
            "source_path",
        ]

    def __init__(self, config_path, on_message):
        Model.__init__(self, config_path, on_message)
        self.device = self.config.get("device", "cuda")
        self.confidence_threshold = float(
            self.config.get("confidence_threshold", 0.5)
        )
        self.source_path = os.path.abspath(
            os.path.expanduser(self.config["source_path"])
        )
        self.checkpoint_path = self.get_model_abs_path(
            self.config, "checkpoint_path"
        )

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "SAM 3.1 checkpoint was not found."
                )
            )
        if not os.path.isdir(self.source_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Official SAM 3.1 source directory was not found."
                )
            )

        self.model = None
        self.marks = []
        self.cache_size = int(self.config.get("cache_size", 2))
        self.image_embedding_cache = LRUCache(self.cache_size)
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False
        self.clip_net = None
        self.classes = []
        self.epsilon = float(self.config.get("epsilon", 0.001))
        self.min_exemplar_bbox_ratio = float(
            self.config.get("min_exemplar_bbox_ratio", 0.45)
        )
        self._load_runtime()

    def _exemplar_reference_area(self):
        """Return the median positive rectangle area, if available."""
        exemplar_areas = []
        for mark in self.marks or []:
            if mark.get("type") != "rectangle" or not bool(
                mark.get("label", 1)
            ):
                continue
            data = mark.get("data", [])
            if len(data) < 4:
                continue
            width = abs(float(data[2]) - float(data[0]))
            height = abs(float(data[3]) - float(data[1]))
            if width > 0 and height > 0:
                exemplar_areas.append(width * height)

        if not exemplar_areas:
            return None
        return float(np.median(exemplar_areas))

    def _filter_shapes_by_exemplar_scale(self, shapes):
        """Drop disconnected polygons much smaller than the exemplars."""
        reference_area = self._exemplar_reference_area()
        if reference_area is None or self.min_exemplar_bbox_ratio <= 0:
            return shapes

        minimum_area = reference_area * self.min_exemplar_bbox_ratio
        kept_shapes = []
        for shape in shapes:
            points = getattr(shape, "points", []) or []
            if not points:
                continue
            cols = [float(point.x()) for point in points]
            rows = [float(point.y()) for point in points]
            bbox_area = float(
                (max(cols) - min(cols) + 1)
                * (max(rows) - min(rows) + 1)
            )
            if bbox_area >= minimum_area:
                kept_shapes.append(shape)

        logger.info(
            "SAM 3.1 exemplar-scale filter kept {}/{} shapes "
            "(minimum bbox area {:.0f})",
            len(kept_shapes),
            len(shapes),
            minimum_area,
        )
        return kept_shapes

    def _load_runtime(self):
        if self.model is not None:
            return
        self.on_message(
            QCoreApplication.translate(
                "Model", "Loading official SAM 3.1 PyTorch model..."
            )
        )
        self.model = OfficialSam31Runtime(
            checkpoint_path=self.checkpoint_path,
            source_path=self.source_path,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        self.stop_inference = False
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        shapes = []
        try:
            self._load_runtime()
            image_embedding = self.image_embedding_cache.get(filename)
            cv_image = qt_img_to_rgb_cv_img(image, filename)
            if image_embedding is None:
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(filename, image_embedding)

            state = self.model.add_language_prompt(image_embedding)
            masks = self.model.predict_masks(
                state,
                self.marks,
                confidence_threshold=self.confidence_threshold,
            )
            for mask in masks:
                mask_2d = mask
                while mask_2d.ndim > 2:
                    mask_2d = mask_2d[0]
                shapes.extend(self.post_process(mask_2d, cv_image))
            shapes = self._filter_shapes_by_exemplar_scale(shapes)
        except Exception as error:  # noqa
            logger.warning("Could not infer with official SAM 3.1")
            logger.warning(error)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        return AutoLabelingResult(shapes, replace=False)

    def unload(self):
        self.stop_inference = True
        self.image_embedding_cache = LRUCache(self.cache_size)
        if self.model is not None:
            self.model.unload()
        self.model = None
        gc.collect()

    def on_next_files_changed(self, next_files):
        # Keep all GPU resources focused on the image currently being edited.
        return
