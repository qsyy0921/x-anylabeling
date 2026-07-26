import base64
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from anylabeling.services.remote_storage import RemoteStorageClient
from anylabeling.views.labeling.label_widget import LabelingWidget


class TestRemoteStorageOverlay(unittest.TestCase):
    def test_render_overlay_posts_revision_and_decodes_png(self):
        png = b"server-rendered-png"
        response = Mock()
        response.json.return_value = {
            "data": {
                "revision": 12,
                "preview_overlay": (
                    "data:image/png;base64,"
                    + base64.b64encode(png).decode("ascii")
                ),
            }
        }

        with (
            patch.dict(
                os.environ,
                {
                    "XANYLABELING_SERVER_URL": "http://server.example",
                    "XANYLABELING_SERVER_API_KEY": "test-token",
                },
            ),
            patch(
                "anylabeling.services.remote_storage.requests.post",
                return_value=response,
            ) as post,
        ):
            revision, overlay = RemoteStorageClient().render_overlay(
                4000,
                3000,
                [
                    {
                        "shape_type": "polygon",
                        "points": [[1, 2], [3, 4]],
                    }
                ],
                12,
            )

        self.assertEqual(revision, 12)
        self.assertEqual(overlay, png)
        self.assertEqual(post.call_args.kwargs["json"]["revision"], 12)
        self.assertEqual(post.call_args.kwargs["json"]["width"], 4000)
        response.raise_for_status.assert_called_once()

    def test_stale_overlay_response_is_ignored(self):
        canvas = SimpleNamespace(set_server_result_overlay=Mock())
        widget = SimpleNamespace(
            _server_overlay_active_request=(8, "current-image"),
            canvas=canvas,
        )

        LabelingWidget._on_server_overlay_sync_finished(
            widget,
            7,
            "previous-image",
            "old-digest",
            b"old-overlay",
            None,
        )

        canvas.set_server_result_overlay.assert_not_called()
        self.assertEqual(
            widget._server_overlay_active_request,
            (8, "current-image"),
        )


if __name__ == "__main__":
    unittest.main()
