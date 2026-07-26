from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import requests


class RemoteStorageClient:
    """Authenticated client for server datasets and staged model uploads."""

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
            params={"path": server_path, "recursive": "true"},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["data"]["files"]

    def sync_dataset(self, server_path: str, cache_root: Path) -> Path:
        files = self.list_files(server_path)
        target_root = cache_root / Path(server_path)
        for item in files:
            relative = Path(item["path"])
            destination = cache_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.suffix.lower() in {".json", ".txt", ".yaml", ".yml"}:
                    # Never overwrite annotations edited on the client.
                    continue
                if destination.stat().st_size == int(item["size"]):
                    continue
            partial = destination.with_suffix(destination.suffix + ".part")
            with requests.get(
                f"{self.server_url}/v1/data/file",
                params={"path": item["path"]},
                headers=self.headers,
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            partial.replace(destination)
        return target_root

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
