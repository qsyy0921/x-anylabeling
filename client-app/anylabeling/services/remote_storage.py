from __future__ import annotations

import base64
import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import requests


class RemoteStorageClient:
    """Authenticated client for server datasets and staged model uploads."""

    URI_SCHEME = "server"

    def __init__(self) -> None:
        self.server_url = os.getenv(
            "XANYLABELING_SERVER_URL", "http://127.0.0.1:18618"
        ).rstrip("/")
        api_key = os.getenv("XANYLABELING_SERVER_API_KEY", "")
        self.headers = {"Token": api_key}
        self.timeout = int(os.getenv("XANYLABELING_SERVER_TIMEOUT", "300"))

    def list_directories(self) -> list[dict]:
        response = requests.get(
            f"{self.server_url}/v1/data/directories",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]["directories"]

    def list_files(self, server_path: str) -> list[dict]:
        response = requests.get(
            f"{self.server_url}/v1/data/files",
            params={
                "path": self.normalize_server_path(server_path),
                "recursive": "true",
            },
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]["files"]

    @classmethod
    def is_server_uri(cls, value: str | os.PathLike | None) -> bool:
        return str(value or "").lower().startswith(f"{cls.URI_SCHEME}://")

    @classmethod
    def normalize_server_path(cls, value: str | os.PathLike) -> str:
        raw = str(value).strip().replace("\\", "/")
        if cls.is_server_uri(raw):
            parsed = urlparse(raw)
            raw = f"{parsed.netloc}{parsed.path}"
        raw = unquote(raw).strip("/")
        data_root = os.getenv(
            "XANYLABELING_SERVER_DATA_ROOT", "/data/mfl/langgao"
        ).replace("\\", "/").rstrip("/")
        if raw == data_root.lstrip("/"):
            return ""
        if raw.startswith(data_root.lstrip("/") + "/"):
            raw = raw[len(data_root.lstrip("/")) + 1 :]
        return PurePosixPath(raw or ".").as_posix().lstrip("./")

    @classmethod
    def server_uri(cls, server_path: str | os.PathLike) -> str:
        return f"{cls.URI_SCHEME}://{cls.normalize_server_path(server_path)}"

    @classmethod
    def label_uri(cls, image_uri: str) -> str:
        path = PurePosixPath(cls.normalize_server_path(image_uri))
        return cls.server_uri(path.with_suffix(".json"))

    def read_file(self, server_path: str) -> bytes:
        response = requests.get(
            f"{self.server_url}/v1/data/file",
            params={"path": self.normalize_server_path(server_path)},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content

    def get_annotation(self, image_path: str) -> dict:
        response = requests.get(
            f"{self.server_url}/v1/data/annotation",
            params={"image_path": self.normalize_server_path(image_path)},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]

    def save_annotation(
        self,
        image_path: str,
        annotation: dict,
        expected_revision: str | None,
    ) -> dict:
        headers = dict(self.headers)
        headers["If-Match"] = expected_revision or "__missing__"
        response = requests.put(
            f"{self.server_url}/v1/data/annotation",
            params={"image_path": self.normalize_server_path(image_path)},
            headers=headers,
            json=annotation,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]

    def delete_annotation(
        self, image_path: str, expected_revision: str | None
    ) -> dict:
        headers = dict(self.headers)
        headers["If-Match"] = expected_revision or "__missing__"
        response = requests.delete(
            f"{self.server_url}/v1/data/annotation",
            params={"image_path": self.normalize_server_path(image_path)},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]

    def upload_model(self, archive_path: Path) -> dict:
        upload_token = os.getenv("XANYLABELING_MODEL_UPLOAD_API_KEY", "")
        if not upload_token:
            raise RuntimeError(
                "This client does not have the server model-upload credential."
            )
        with archive_path.open("rb") as source:
            response = requests.post(
                f"{self.server_url}/v1/models/upload",
                headers={
                    **self.headers,
                    "X-Model-Upload-Token": upload_token,
                },
                files={"file": (archive_path.name, source, "application/zip")},
                timeout=max(self.timeout, 1800),
            )
        response.raise_for_status()
        return response.json()["data"]

    def list_model_registry(self) -> list[dict]:
        response = requests.get(
            f"{self.server_url}/v1/model-registry",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]["models"]

    def render_overlay(
        self,
        width: int,
        height: int,
        shapes: list[dict],
        revision: int,
    ) -> tuple[int, bytes]:
        """Ask the server to rasterize the latest annotation state."""
        response = requests.post(
            f"{self.server_url}/v1/render-overlay",
            headers=self.headers,
            json={
                "width": width,
                "height": height,
                "shapes": shapes,
                "revision": revision,
            },
            timeout=min(self.timeout, 60),
        )
        response.raise_for_status()
        data = response.json()["data"]
        overlay = data["preview_overlay"]
        if "," in overlay:
            overlay = overlay.split(",", 1)[1]
        return int(data["revision"]), base64.b64decode(overlay)

    def install_registry_model(self, model_id: str) -> dict:
        response = requests.post(
            f"{self.server_url}/v1/model-registry/{model_id}/install",
            headers={
                **self.headers,
                "X-Model-Upload-Token": self._model_admin_token(),
            },
            timeout=max(self.timeout, 7200),
        )
        response.raise_for_status()
        return response.json()["data"]

    def enable_registry_model(self, model_id: str) -> dict:
        response = requests.post(
            f"{self.server_url}/v1/model-registry/{model_id}/enable",
            headers={
                **self.headers,
                "X-Model-Upload-Token": self._model_admin_token(),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]

    @staticmethod
    def _model_admin_token() -> str:
        token = os.getenv("XANYLABELING_MODEL_UPLOAD_API_KEY", "")
        if not token:
            raise RuntimeError(
                "This client does not have the server model-management credential."
            )
        return token
