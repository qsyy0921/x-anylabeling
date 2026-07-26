import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.schemas.request import PredictRequest, RenderOverlayRequest
from app.schemas.response import (
    ErrorDetail,
    ErrorResponse,
    PredictResponse,
    SuccessResponse,
)

router = APIRouter()


def _render_preview_overlay_for_size(width, height, shapes):
    """Render vector annotations as a transparent full-resolution PNG."""
    if width <= 0 or height <= 0 or width * height > 64_000_000:
        raise ValueError("Invalid overlay dimensions")

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    fill_color = (82, 211, 164, 72)
    line_color = (44, 181, 132, 235)
    line_width = max(2, int(round(max(height, width) / 1500)))

    for shape in shapes:
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        pts = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        shape_type = shape.get("shape_type", "polygon")

        if shape_type == "rectangle" and len(pts) >= 2:
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
            cv2.rectangle(
                overlay, (x1, y1), (x2, y2), line_color, line_width
            )
        elif len(pts) >= 3:
            polygon = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [polygon], fill_color)
            cv2.polylines(
                overlay, [polygon], True, line_color, line_width, cv2.LINE_AA
            )

    encoded, buffer = cv2.imencode(
        ".png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not encoded:
        raise RuntimeError("Failed to encode server preview overlay")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _render_preview_overlay(image, shapes):
    """Render vector predictions once on the server as a transparent PNG."""
    height, width = image.shape[:2]
    return _render_preview_overlay_for_size(width, height, shapes)


@router.post("/v1/render-overlay")
async def render_overlay(request: RenderOverlayRequest):
    """Redraw edited annotations without loading or running an AI model."""
    try:
        overlay = _render_preview_overlay_for_size(
            request.width,
            request.height,
            request.shapes,
        )
        return SuccessResponse(
            data={
                "revision": request.revision,
                "preview_overlay": overlay,
                "preview_shape_count": len(request.shapes),
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Overlay rendering error: {e}")
        return ErrorResponse(
            error=ErrorDetail(code="OVERLAY_RENDER_ERROR", message=str(e))
        )


@router.post("/v1/predict")
async def predict(request: PredictRequest):
    """Execute prediction on an image.

    Args:
        request: Prediction request with model, image, and parameters.

    Returns:
        Success response with prediction results or error response.
    """
    from app.main import inference_executor, loader

    try:
        _ = loader.get_model(request.model)
    except ValueError as e:
        return ErrorResponse(
            error=ErrorDetail(code="MODEL_NOT_FOUND", message=str(e))
        )

    try:
        if request.server_image_path:
            from app.api.storage import _resolve_under

            data_root = Path(
                os.getenv("AUTOLABEL_DATA_ROOT", "/data/mfl/langgao")
            ).expanduser().resolve()
            server_image = _resolve_under(
                data_root, request.server_image_path
            )
            image = cv2.imread(str(server_image), cv2.IMREAD_COLOR)
        else:
            image_data = (
                request.image.split(",")[1]
                if "," in request.image
                else request.image
            )
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_IMAGE", message="Failed to decode image"
                )
            )
    except Exception as e:
        logger.error(f"Image decoding error: {e}")
        return ErrorResponse(
            error=ErrorDetail(
                code="INVALID_IMAGE",
                message=f"Failed to decode image: {str(e)}",
            )
        )

    try:
        params = dict(request.params)
        render_preview = bool(params.pop("_render_preview", False))
        result = await inference_executor.execute(request.model, image, params)
        if render_preview:
            shapes = result.get("shapes", [])
            result["preview_overlay"] = _render_preview_overlay(image, shapes)
            result["preview_shape_count"] = len(shapes)
        return SuccessResponse(data=PredictResponse(**result))
    except RuntimeError as e:
        if "queue is full" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task queue is full, please try again later",
            )
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return ErrorResponse(
            error=ErrorDetail(code="INFERENCE_ERROR", message=str(e))
        )
