import asyncio
import base64

import cv2
import numpy as np

from app.api.predict import _render_preview_overlay, render_overlay
from app.schemas.request import RenderOverlayRequest


def test_render_preview_overlay_preserves_image_size_and_alpha():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    shapes = [
        {
            "shape_type": "rectangle",
            "points": [[10, 10], [80, 60]],
        },
        {
            "shape_type": "polygon",
            "points": [[100, 20], [180, 20], [140, 90]],
        },
    ]

    data_uri = _render_preview_overlay(image, shapes)
    encoded = base64.b64decode(data_uri.split(",", 1)[1])
    overlay = cv2.imdecode(
        np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED
    )

    assert data_uri.startswith("data:image/png;base64,")
    assert overlay.shape == (120, 200, 4)
    assert overlay[0, 0, 3] == 0
    assert overlay[:, :, 3].max() > 0


def test_render_overlay_echoes_revision_and_removes_deleted_shape():
    two_shapes = [
        {
            "label": "first",
            "shape_type": "rectangle",
            "points": [[10, 10], [40, 40]],
        },
        {
            "label": "second",
            "shape_type": "rectangle",
            "points": [[60, 10], [90, 40]],
        },
    ]

    first = asyncio.run(
        render_overlay(
            RenderOverlayRequest(
                width=100,
                height=60,
                shapes=two_shapes,
                revision=7,
            )
        )
    )
    second = asyncio.run(
        render_overlay(
            RenderOverlayRequest(
                width=100,
                height=60,
                shapes=two_shapes[:1],
                revision=8,
            )
        )
    )

    assert first.data["revision"] == 7
    assert first.data["preview_shape_count"] == 2
    assert second.data["revision"] == 8
    assert second.data["preview_shape_count"] == 1
    assert first.data["preview_overlay"] != second.data["preview_overlay"]
