import os
from io import BytesIO
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

from anylabeling.services.remote_storage import RemoteStorageClient
from anylabeling.views.labeling.label_file import LabelFile


def test_server_path_normalization_and_uri_round_trip():
    with patch.dict(
        os.environ,
        {"XANYLABELING_SERVER_DATA_ROOT": "/data/mfl/langgao"},
    ):
        assert (
            RemoteStorageClient.normalize_server_path(
                "/data/mfl/langgao/project/images"
            )
            == "project/images"
        )
        uri = RemoteStorageClient.server_uri("project/images/cam0.jpg")
        assert uri == "server://project/images/cam0.jpg"
        assert RemoteStorageClient.normalize_server_path(uri) == (
            "project/images/cam0.jpg"
        )
        assert RemoteStorageClient.label_uri(uri) == (
            "server://project/images/cam0.json"
        )


def test_save_annotation_uses_missing_revision_guard():
    response = Mock()
    response.json.return_value = {"data": {"revision": "new-revision"}}
    with patch(
        "anylabeling.services.remote_storage.requests.put",
        return_value=response,
    ) as put:
        result = RemoteStorageClient().save_annotation(
            "server://dataset/image.jpg",
            {"imagePath": "image.jpg", "shapes": []},
            None,
        )

    assert result["revision"] == "new-revision"
    assert put.call_args.kwargs["headers"]["If-Match"] == "__missing__"
    response.raise_for_status.assert_called_once()


def test_label_file_loads_annotation_from_memory():
    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    image_data = buffer.getvalue()
    label_file = LabelFile()
    label_file.load_data(
        {
            "version": "2.5.4",
            "flags": {},
            "shapes": [],
            "imagePath": "image.png",
            "imageData": None,
            "imageHeight": 1,
            "imageWidth": 1,
        },
        image_data=image_data,
        filename="server://dataset/image.json",
    )

    assert label_file.image_data == image_data
    assert label_file.image_path == "image.png"
    assert label_file.shapes == []
