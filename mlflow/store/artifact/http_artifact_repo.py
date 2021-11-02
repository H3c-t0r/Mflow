import os
import requests
import posixpath

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.utils.file_utils import relative_path_to_artifact_path


class HttpArtifactRepository(ArtifactRepository):
    """Stores artifacts in a remote artifact storage using HTTP requests"""

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)

        file_name = os.path.basename(local_file)
        paths = (artifact_path, file_name) if artifact_path else (file_name,)
        with open(local_file, "rb") as f:
            url = posixpath.join(self.artifact_uri, *paths)
            resp = requests.put(url, data=f)
            resp.raise_for_status()

    def log_artifacts(self, local_dir, artifact_path=None):
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            if root == local_dir:
                artifact_dir = artifact_path
            else:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_dir = (
                    posixpath.join(artifact_path, rel_path) if artifact_path else rel_path
                )
            for f in filenames:
                self.log_artifact(os.path.join(root, f), artifact_dir)

    def list_artifacts(self, path=None):
        sep = "/mlflow-artifacts/artifacts"
        head, tail = self.artifact_uri.split(sep, maxsplit=1)
        url = head + sep
        tail = tail.lstrip("/")
        params = {"path": posixpath.join(tail, path) if path else tail}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        json = resp.json()
        files = json.get("files", [])
        return sorted(
            [
                FileInfo(
                    posixpath.join(path, f["path"]) if path else f["path"],
                    f["is_dir"],
                    int(f.get("file_size")),
                )
                for f in files
            ],
            key=lambda f: f.path,
        )

    def _download_file(self, remote_file_path, local_path):
        url = posixpath.join(self.artifact_uri, remote_file_path)
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
