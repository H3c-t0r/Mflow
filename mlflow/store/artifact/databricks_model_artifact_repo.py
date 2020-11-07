import logging
import json

import mlflow.tracking
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException

from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import download_file_using_signed_uri
from mlflow.utils.rest_utils import http_request
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_profile,
    parse_model_uri,
    get_model_version_from_stage,
)

_logger = logging.getLogger(__name__)
_DOWNLOAD_CHUNK_SIZE = 100000000
REGISTRY_LIST_ARTIFACTS_ENDPOINT = "/api/2.0/mlflow/model-versions/list-artifacts"
REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT = "/api/2.0/mlflow/model-versions/get-signed-download-uri"


class DatabricksModelArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on model registry artifacts in the access-controlled
    `dbfs:/databricks/model-registry` location

    Signed access URIs for S3 / Azure Blob Storage are fetched from the MLflow service and used to
    download model artifacts.

    The artifact_uri is expected to be of the form
    - `models:/<model_name>/<model_version>`
    - `models:/<model_name>/<stage>`  (refers to the latest model version in the given stage)
    - `models://<profile>/<model_name>/<model_version or state>`

    Note : This artifact repository is meant is to be instantiated by the ModelsArtifactRepository
    when the model download uri is of the form
    `dbfs:/databricks/mlflow-registry/<model-version-id>/models/<artifact-path>`
    """

    def __init__(self, artifact_uri):
        if not is_databricks_profile(artifact_uri):
            raise MlflowException(
                message="A valid databricks profile is required to use this repository",
                error_code=INVALID_PARAMETER_VALUE,
            )
        super().__init__(artifact_uri)
        from mlflow.tracking import MlflowClient

        databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(artifact_uri) or mlflow.get_registry_uri()
        )

        self.client = MlflowClient(registry_uri=databricks_profile_uri)
        hostcreds_from_uri = get_databricks_host_creds(databricks_profile_uri)
        self.get_host_creds = lambda: hostcreds_from_uri
        self.model_name, self.model_version = self._get_model_name_and_version(artifact_uri)

    def _get_model_name_and_version(self, artifact_uri):
        (model_name, model_version, model_stage) = parse_model_uri(artifact_uri)
        if model_stage is not None:
            model_version = get_model_version_from_stage(self.client, model_name, model_stage)
        return model_name, str(model_version)

    def _call_endpoint(self, json, endpoint):
        host_creds = self.get_host_creds()
        return http_request(host_creds=host_creds, endpoint=endpoint, method="GET", params=json)

    def _make_json_body(self, path, page_token=None):
        if not page_token:
            return {"name": self.model_name, "version": self.model_version, "path": path}
        return {
            "name": self.model_name,
            "version": self.model_version,
            "path": path,
            "page_token": page_token,
        }

    def list_artifacts(self, path=None):
        infos = []
        page_token = None
        if not path:
            path = ""
        while True:
            json_body = self._make_json_body(path, page_token)
            response = self._call_endpoint(json_body, REGISTRY_LIST_ARTIFACTS_ENDPOINT)
            try:
                json_response = json.loads(response.text)
            except ValueError:
                raise MlflowException(
                    "API request to list files under path `%s` failed with status code %s. "
                    "Response body: %s" % (path, response.status_code, response.text)
                )
            artifact_list = json_response.get("files", [])
            next_page_token = json_response.get("next_page_token", None)
            # If `path` is a file, ListArtifacts returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if (
                len(artifact_list) == 1
                and artifact_list[0]["path"] == path
                and not artifact_list[0]["is_dir"]
            ):
                return []
            for output_file in artifact_list:
                artifact_size = None if output_file["is_dir"] else output_file["file_size"]
                infos.append(FileInfo(output_file["path"], output_file["is_dir"], artifact_size))
            if len(artifact_list) == 0 or not next_page_token:
                break
            page_token = next_page_token
        return infos

    def _get_signed_download_uri(self, path=None):
        if not path:
            path = ""
        json_body = self._make_json_body(path)
        response = self._call_endpoint(json_body, REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT)
        try:
            json_response = json.loads(response.text)
        except ValueError:
            raise MlflowException(
                "API request to get presigned uri to for file under path `%s` failed with"
                " status code %s. Response body: %s" % (path, response.status_code, response.text)
            )
        return json_response.get("signed_uri", None)

    def _download_file(self, remote_file_path, local_path):
        try:
            signed_uri = self._get_signed_download_uri(remote_file_path)
            download_file_using_signed_uri(signed_uri, local_path, _DOWNLOAD_CHUNK_SIZE)
        except Exception as err:
            raise MlflowException(err)

    def log_artifact(self, local_file, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def log_artifacts(self, local_dir, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
