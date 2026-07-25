import numpy as np
from loguru import logger
from PIL import Image
from typing import Any, Dict, List

from . import BaseModel
from app.schemas.shape import Shape
from app.core.registry import register_model


@register_model("paddleocr_vl_1_5")
class PaddleOCRVL15(BaseModel):
    """PaddleOCR-VL-1.5 unified OCR model supporting multiple tasks."""

    def load(self):
        """Load PaddleOCR-VL-1.5 model and initialize components."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = self.params.get(
            "model_path", "PaddlePaddle/PaddleOCR-VL-1.5"
        )
        device = self.params.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch_dtype = getattr(
            torch, self.params.get("torch_dtype", "bfloat16")
        )

        self.device = device
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch_dtype
            )
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.prompts = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }

        logger.info(f"PaddleOCR-VL-1.5 model loaded on {device}")

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata including available tasks."""
        metadata = super().get_metadata()
        metadata["available_tasks"] = [
            {
                "id": "ocr",
                "name": "OCR",
                "description": "Optical Character Recognition for text extraction",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "table",
                "name": "Table Recognition",
                "description": "Extract table structure and content",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "formula",
                "name": "Formula Recognition",
                "description": "Recognize mathematical formulas",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "chart",
                "name": "Chart Recognition",
                "description": "Extract information from charts and graphs",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "spotting",
                "name": "Text Spotting",
                "description": "Detect and recognize text with bounding boxes",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
                    "toggle_preserve_existing_annotations": {},
                },
            },
            {
                "id": "seal",
                "name": "Seal Recognition",
                "description": "Recognize seal stamps and chop marks",
                "batch_processing_mode": "default",
                "active_widgets": {
                    "button_run": {},
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
            params: Inference parameters including task type.

        Returns:
            Dictionary with prediction results.
        """
        task = params.get("current_task", "ocr")
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")

        if task == "spotting":
            return self._predict_spotting(pil_image)
        else:
            return self._predict_text_task(pil_image, task)

    def _predict_text_task(
        self, image: Image.Image, task: str
    ) -> Dict[str, Any]:
        """Execute text-based recognition tasks.

        Args:
            image: Input image in PIL format.
            task: Task type (ocr, table, formula, chart, seal).

        Returns:
            Dictionary with text description.
        """
        if task not in self.prompts:
            logger.warning(f"Unsupported task: {task}, falling back to OCR")
            task = "ocr"

        max_pixels = self.params.get("max_pixels", 1280 * 28 * 28)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompts[task]},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(self.device)

        max_new_tokens = self.params.get("max_new_tokens", 512)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        result = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] : -1]
        )

        return {"shapes": [], "description": result}

    def _predict_spotting(self, image: Image.Image) -> Dict[str, Any]:
        """Execute text spotting task with bounding boxes.

        Args:
            image: Input image in PIL format.

        Returns:
            Dictionary with detected text shapes.
        """
        orig_w, orig_h = image.size
        spotting_upscale_threshold = self.params.get(
            "spotting_upscale_threshold", 1500
        )

        process_image = image
        if (
            orig_w < spotting_upscale_threshold
            and orig_h < spotting_upscale_threshold
        ):
            process_w, process_h = orig_w * 2, orig_h * 2
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            process_image = image.resize(
                (process_w, process_h), resample_filter
            )

        max_pixels = self.params.get("spotting_max_pixels", 2048 * 28 * 28)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": process_image},
                    {"type": "text", "text": self.prompts["spotting"]},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(self.device)

        max_new_tokens = self.params.get("max_new_tokens", 512)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        result = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1] : -1]
        )
        logger.debug(f"Spotting result: {result}")

        shapes = self._parse_spotting_result(result, orig_w, orig_h)

        return {"shapes": shapes, "description": ""}

    def _parse_spotting_result(
        self,
        result: str,
        orig_w: int,
        orig_h: int,
    ) -> List[Shape]:
        """Parse spotting result string into Shape objects.

        Args:
            result: Model output string containing bounding box coordinates.
            orig_w: Original image width.
            orig_h: Original image height.

        Returns:
            List of Shape objects with text and bounding boxes.
        """
        import re

        shapes = []

        lines = result.strip().split('\n')
        for line in lines:
            if not line.strip() or '<|LOC_' not in line:
                continue

            try:
                loc_pattern = r'<\|LOC_(\d+)\|>'
                matches = re.findall(loc_pattern, line)

                if len(matches) < 8:
                    continue

                text_match = re.match(r'^(.*?)<\|LOC_', line)
                text = text_match.group(1).strip() if text_match else ""

                if not text:
                    continue

                normalized_coords = [int(m) / 1000.0 for m in matches[:8]]

                points = []
                for i in range(0, 8, 2):
                    x_norm = normalized_coords[i]
                    y_norm = normalized_coords[i + 1]

                    x_orig = x_norm * orig_w
                    y_orig = y_norm * orig_h

                    points.append([float(x_orig), float(y_orig)])

                shape = Shape(
                    label="text",
                    shape_type="polygon",
                    points=points,
                    description=text,
                )
                shapes.append(shape)
            except (ValueError, IndexError, AttributeError) as e:
                logger.error(f"Failed to parse spotting result: {e}")
                continue

        return shapes

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
