import numpy as np
from loguru import logger
from PIL import Image
from typing import Any, Dict, List

from . import BaseModel, parse_prompts
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("rexomni")
class RexOmni(BaseModel):
    """Rex-Omni unified vision fundation model supporting multiple tasks."""

    def load(self):
        """Load RexOmni model and initialize components."""
        import sys
        import os

        rex_omni_parent_dir = os.path.join(os.path.dirname(__file__))
        if rex_omni_parent_dir not in sys.path:
            sys.path.insert(0, rex_omni_parent_dir)

        from rex_omni.wrapper import RexOmniWrapper
        from rex_omni.tasks import TaskType

        model_path = self.params.get("model_path")
        if not model_path:
            raise ValueError("model_path is required in params")

        backend = self.params.get("backend", "transformers")
        max_tokens = self.params.get("max_tokens", 4096)
        attn_implementation = self.params.get("attn_implementation", None)
        device_map = self.params.get("device_map", "auto")

        import torch

        init_kwargs = {
            "trust_remote_code": True,
        }

        if backend == "transformers":
            init_kwargs.update(
                {
                    "device_map": device_map,
                    "torch_dtype": torch.bfloat16,
                }
            )

        elif backend == "vllm":
            init_kwargs.update(
                {
                    "tokenizer_mode": self.params.get(
                        "tokenizer_mode", "slow"
                    ),
                    "limit_mm_per_prompt": self.params.get(
                        "limit_mm_per_prompt", {"image": 1, "video": 0}
                    ),
                    "max_model_len": self.params.get("max_model_len", 4096),
                    "gpu_memory_utilization": self.params.get(
                        "gpu_memory_utilization", 0.8
                    ),
                    "tensor_parallel_size": self.params.get(
                        "tensor_parallel_size", 1
                    ),
                    "quantization": self.params.get("quantization", None),
                }
            )

        self.model = RexOmniWrapper(
            model_path=model_path,
            backend=backend,
            max_tokens=max_tokens,
            temperature=self.params.get("temperature", 0.0),
            top_p=self.params.get("top_p", 0.8),
            top_k=self.params.get("top_k", 1),
            repetition_penalty=self.params.get("repetition_penalty", 1.05),
            min_pixels=self.params.get("min_pixels", 16 * 28 * 28),
            max_pixels=self.params.get("max_pixels", 2560 * 28 * 28),
            attn_implementation=attn_implementation,
            **init_kwargs,
        )
        self.task_type = TaskType

        logger.info(f"RexOmni model loaded with {backend} backend")

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata including available tasks."""
        metadata = super().get_metadata()
        metadata["available_tasks"] = [
            {
                "id": "detection",
                "name": "Detection",
                "description": "Detect objects and return bounding boxes",
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {},
                    "button_send": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "keypoint_person",
                "name": "Keypoint (Person)",
                "description": "Detect keypoints with skeleton visualization",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "keypoint_animal",
                "name": "Keypoint (Animal)",
                "description": "Detect keypoints with skeleton visualization",
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {
                        "placeholder": "Enter a single animal category here...",
                        "tooltip": "Enter a single animal category (e.g., 'cat', 'dog', 'horse') or 'animal' to detect all animal types",
                    },
                    "button_send": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "keypoint_hand",
                "name": "Keypoint (Hand)",
                "description": "Detect hand keypoints with skeleton visualization",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "ocr_box_word",
                "name": "OCR Box (Word Level)",
                "description": "Word-level text detection and recognition in boxes",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "ocr_box_text_line",
                "name": "OCR Box (Text Line Level)",
                "description": "Text line-level text detection and recognition in boxes",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "ocr_polygon_word",
                "name": "OCR Polygon (Word Level)",
                "description": "Word-Level text detection and recognition in polygons",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "ocr_polygon_text_line",
                "name": "OCR Polygon (Text Line Level)",
                "description": "Text line-Level text detection and recognition in polygons",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "pointing",
                "name": "Pointing",
                "description": "Point to objects based on description",
                "batch_processing_mode": "text_prompt",
                "active_widgets": {
                    "edit_text": {
                        "placeholder": "Enter object description here...",
                        "tooltip": (
                            "Describe the object or location you want to point to. Examples: "
                            "'appele', "
                            "'open boxes,closed boxes', "
                            "'the person wearing a blue shirt'"
                        ),
                    },
                    "button_send": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "visual_prompting",
                "name": "Visual Prompting",
                "description": "Ground visual examples to find similar objects",
                "batch_processing_mode": None,
                "active_widgets": {
                    "add_pos_rect": {},
                    "button_run_rect": {},
                    "button_clear": {},
                    "button_finish_object": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
        ]
        return metadata

    def predict(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute prediction based on task type.

        Args:
            image: Input image in BGR format.
            params: Inference parameters including task type and categories.

        Returns:
            Dictionary with prediction results.
        """
        task = params.get("current_task", "detection")
        text_prompt = params.get("text_prompt", "")

        pil_image = Image.fromarray(image[:, :, ::-1])

        if task == "detection":
            return self._predict_detection(pil_image, text_prompt)
        elif task in ["keypoint_person", "keypoint_animal"]:
            keypoint_type = "person" if task == "keypoint_person" else "animal"
            return self._predict_keypoint(
                pil_image, keypoint_type, text_prompt
            )
        elif task == "keypoint_hand":
            return self._predict_keypoint_hand(pil_image)
        elif task in ["ocr_box_word", "ocr_box_text_line"]:
            if task == "ocr_box_word":
                categories = ["word"]
            else:
                categories = ["text line"]
            return self._predict_ocr_box(pil_image, categories)
        elif task in ["ocr_polygon_word", "ocr_polygon_text_line"]:
            if task == "ocr_polygon_word":
                categories = ["word"]
            else:
                categories = ["text line"]
            return self._predict_ocr_polygon(pil_image, categories)
        elif task == "pointing":
            return self._predict_pointing(pil_image, text_prompt)
        elif task == "visual_prompting":
            marks = params.get("marks", [])
            return self._predict_visual_prompting(pil_image, marks)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _predict_detection(
        self, image: Image.Image, text_prompt: str
    ) -> Dict[str, Any]:
        """Execute object detection task."""
        if not text_prompt:
            logger.warning("Please provide categories for detection task.")
            return {"shapes": [], "description": ""}

        categories = parse_prompts(text_prompt)

        results = self.model.inference(
            images=image, task=self.task_type.DETECTION, categories=categories
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        shapes = self._convert_detection_to_shapes(extracted)

        return {"shapes": shapes, "description": ""}

    def _predict_keypoint(
        self, image: Image.Image, keypoint_type: str, text_prompt: str = ""
    ) -> Dict[str, Any]:
        """Execute keypoint detection task."""
        if keypoint_type == "person":
            categories = ["person"]
        else:  # animal
            if not text_prompt:
                logger.warning(
                    "Please provide categories for animal keypoint task."
                )
                return {"shapes": [], "description": ""}
            else:
                categories = parse_prompts(text_prompt)

        results = self.model.inference(
            images=image,
            task=self.task_type.KEYPOINT,
            categories=categories,
            keypoint_type=keypoint_type,
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        shapes = self._convert_keypoint_to_shapes(extracted)

        return {"shapes": shapes, "description": ""}

    def _predict_keypoint_hand(self, image: Image.Image) -> Dict[str, Any]:
        """Execute hand keypoint detection task."""
        results = self.model.inference(
            images=image,
            task=self.task_type.KEYPOINT,
            categories=["hand"],
            keypoint_type="hand",
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        shapes = self._convert_keypoint_to_shapes(extracted)

        return {"shapes": shapes, "description": ""}

    def _predict_ocr_box(
        self, image: Image.Image, categories: List
    ) -> Dict[str, Any]:
        """Execute OCR box detection task."""
        results = self.model.inference(
            images=image, task=self.task_type.OCR_BOX, categories=categories
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        shapes = self._convert_ocr_to_shapes(
            extracted, categories, shape_type="rectangle"
        )

        return {"shapes": shapes, "description": ""}

    def _predict_ocr_polygon(
        self, image: Image.Image, categories: List
    ) -> Dict[str, Any]:
        """Execute OCR polygon detection task."""
        results = self.model.inference(
            images=image,
            task=self.task_type.OCR_POLYGON,
            categories=categories,
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        shapes = self._convert_ocr_to_shapes(
            extracted, categories, shape_type="quadrilateral"
        )

        return {"shapes": shapes, "description": ""}

    def _predict_pointing(
        self, image: Image.Image, text_prompt: str
    ) -> Dict[str, Any]:
        """Execute pointing task."""
        if not text_prompt:
            logger.warning("Please provide a prompt for pointing task.")
            return {"shapes": [], "description": ""}

        categories = parse_prompts(text_prompt)

        results = self.model.inference(
            images=image, task=self.task_type.POINTING, categories=categories
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        shapes = self._convert_pointing_to_shapes(extracted)

        return {"shapes": shapes, "description": ""}

    def _predict_visual_prompting(
        self, image: Image.Image, marks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute visual prompting task based on reference boxes.

        Args:
            image: Input image in PIL format.
            marks: List of visual prompts with type, label, and data fields.
                Each mark should have:
                - type: "rectangle" for bounding box prompts
                - label: 1 for positive prompts, 0 for negative prompts
                - data: [x1, y1, x2, y2] coordinates

        Returns:
            Dictionary with shapes and description.
        """
        if not marks:
            logger.warning(
                "Please provide reference boxes for visual prompting task."
            )
            return {"shapes": [], "description": ""}

        visual_prompt_boxes = []
        for mark in marks:
            if mark.get("type") != "rectangle":
                continue

            label = mark.get("label", 0)
            if label != 1:
                continue

            data = mark.get("data", [])
            if len(data) != 4:
                continue

            x1, y1, x2, y2 = data
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            visual_prompt_boxes.append([x_min, y_min, x_max, y_max])

        if not visual_prompt_boxes:
            logger.warning(
                "No valid positive reference boxes provided for visual prompting."
            )
            return {"shapes": [], "description": ""}

        logger.info(
            f"Processing visual prompting with {len(visual_prompt_boxes)} reference boxes"
        )

        results = self.model.inference(
            images=image,
            task=self.task_type.VISUAL_PROMPTING,
            visual_prompt_boxes=visual_prompt_boxes,
        )

        if not results or not results[0].get("success"):
            return {"shapes": [], "description": ""}

        result = results[0]
        extracted = result.get("extracted_predictions", {})
        extracted_with_label = {"AUTOLABEL_OBJECT": []}
        for annotations in extracted.values():
            extracted_with_label["AUTOLABEL_OBJECT"].extend(annotations)

        shapes = self._convert_detection_to_shapes(extracted_with_label)

        return {"shapes": shapes, "description": ""}

    def _convert_detection_to_shapes(
        self, extracted: Dict[str, List]
    ) -> List[Shape]:
        """Convert detection predictions to Shape objects."""
        shapes = []

        for category, annotations in extracted.items():
            for ann in annotations:
                if ann.get("type") == "box":
                    coords = ann.get("coords", [])
                    if len(coords) == 4:
                        x0, y0, x1, y1 = coords
                        shape = Shape(
                            label=category,
                            shape_type="rectangle",
                            points=[
                                [float(x0), float(y0)],
                                [float(x1), float(y0)],
                                [float(x1), float(y1)],
                                [float(x0), float(y1)],
                            ],
                        )
                        shapes.append(shape)

        return shapes

    def _convert_keypoint_to_shapes(
        self, extracted: Dict[str, List]
    ) -> List[Shape]:
        """Convert keypoint predictions to Shape objects."""
        shapes = []

        group_id = 1
        for category, instances in extracted.items():
            for instance in instances:
                if instance.get("type") != "keypoint":
                    continue

                bbox = instance.get("bbox", [])
                keypoints = instance.get("keypoints", {})

                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = bbox
                    bbox_shape = Shape(
                        label=category,
                        shape_type="rectangle",
                        points=[
                            [float(x0), float(y0)],
                            [float(x1), float(y0)],
                            [float(x1), float(y1)],
                            [float(x0), float(y1)],
                        ],
                        group_id=group_id,
                    )
                    shapes.append(bbox_shape)

                for kp_name, kp_coords in keypoints.items():
                    if kp_coords == "unvisible" or not isinstance(
                        kp_coords, list
                    ):
                        continue
                    if len(kp_coords) == 2:
                        x, y = kp_coords
                        kp_shape = Shape(
                            label=kp_name,
                            shape_type="point",
                            points=[[float(x), float(y)]],
                            group_id=group_id,
                        )
                        shapes.append(kp_shape)

                group_id += 1

        return shapes

    def _convert_ocr_to_shapes(
        self, extracted: Dict[str, List], categories: List, shape_type: str
    ) -> List[Shape]:
        """Convert OCR predictions to Shape objects."""
        shapes = []

        for category, annotations in extracted.items():
            for ann in annotations:
                ann_type = ann.get("type")
                coords = ann.get("coords", [])

                if ann_type == "box" and shape_type == "rectangle":
                    if len(coords) == 4:
                        x0, y0, x1, y1 = coords
                        text = ann.get("text", category)
                        shape = Shape(
                            label=categories[0],
                            shape_type="rectangle",
                            points=[
                                [float(x0), float(y0)],
                                [float(x1), float(y0)],
                                [float(x1), float(y1)],
                                [float(x0), float(y1)],
                            ],
                            description=text,
                        )
                        shapes.append(shape)
                elif ann_type == "polygon" and shape_type == "quadrilateral":
                    if len(coords) >= 3:
                        points = [[float(p[0]), float(p[1])] for p in coords]
                        text = ann.get("text", category)
                        shape = Shape(
                            label=categories[0],
                            shape_type="quadrilateral",
                            points=points,
                            description=text,
                        )
                        shapes.append(shape)

        return shapes

    def _convert_pointing_to_shapes(
        self, extracted: Dict[str, List]
    ) -> List[Shape]:
        """Convert pointing predictions to Shape objects."""
        shapes = []

        for category, annotations in extracted.items():
            for ann in annotations:
                if ann.get("type") == "point":
                    coords = ann.get("coords", [])
                    if len(coords) == 2:
                        x, y = coords
                        shape = Shape(
                            label=category,
                            shape_type="point",
                            points=[[float(x), float(y)]],
                        )
                        shapes.append(shape)

        return shapes

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
