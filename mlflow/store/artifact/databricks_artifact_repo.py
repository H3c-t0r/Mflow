import base64
import logging
import os
import posixpath
import requests
import uuid
import tempfile
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from mlflow.azure.client import put_block, put_block_list
import mlflow.tracking
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INTERNAL_ERROR,
    RESOURCE_DOES_NOT_EXIST,
)

from mlflow.protos.databricks_artifacts_pb2 import (
    DatabricksMlflowArtifactsService,
    GetCredentialsForWrite,
    GetCredentialsForRead,
    ArtifactCredentialType,
)
from mlflow.protos.service_pb2 import MlflowService, GetRun, ListArtifacts
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import (
    download_file_using_http_uri,
    relative_path_to_artifact_path,
    yield_file_in_chunks,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils import rest_utils
from mlflow.utils.rest_utils import (
    call_endpoint,
    extract_api_info_for_service,
    _REST_API_PATH_PREFIX,
)
from mlflow.utils.uri import (
    extract_and_normalize_path,
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_acled_artifacts_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
)

_logger = logging.getLogger(__name__)
_AZURE_MAX_BLOCK_CHUNK_SIZE = 100000000  # Max. size of each block allowed is 100 MB in stage_block
_DOWNLOAD_CHUNK_SIZE = 100000000
_SERVICE_AND_METHOD_TO_INFO = {
    service: extract_api_info_for_service(service, _REST_API_PATH_PREFIX)
    for service in [MlflowService, DatabricksMlflowArtifactsService]
}


class DatabricksArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on artifacts in the access-controlled
    `dbfs:/databricks/mlflow-tracking` location.

    Signed access URIs for S3 / Azure Blob Storage are fetched from the MLflow service and used to
    read and write files from/to this location.

    The artifact_uri is expected to be of the form
    dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/
    """

    def __init__(self, artifact_uri):
        if not is_valid_dbfs_uri(artifact_uri):
            raise MlflowException(
                message="DBFS URI must be of the form dbfs:/<path> or "
                + "dbfs://profile@databricks/<path>",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not is_databricks_acled_artifacts_uri(artifact_uri):
            raise MlflowException(
                message=(
                    "Artifact URI incorrect. Expected path prefix to be"
                    " databricks/mlflow-tracking/path/to/artifact/.."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        # The dbfs:/ path ultimately used for artifact operations should not contain the
        # Databricks profile info, so strip it before setting ``artifact_uri``.
        super().__init__(remove_databricks_profile_info_from_artifact_uri(artifact_uri))

        self.databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(artifact_uri)
            or mlflow.tracking.get_tracking_uri()
        )
        self.run_id = self._extract_run_id(self.artifact_uri)
        # Fetch the artifact root for the MLflow Run associated with `artifact_uri` and compute
        # the path of `artifact_uri` relative to the MLflow Run's artifact root
        # (the `run_relative_artifact_repo_root_path`). All operations performed on this artifact
        # repository will be performed relative to this computed location
        artifact_repo_root_path = extract_and_normalize_path(artifact_uri)
        run_artifact_root_uri = self._get_run_artifact_root(self.run_id)
        run_artifact_root_path = extract_and_normalize_path(run_artifact_root_uri)
        run_relative_root_path = posixpath.relpath(
            path=artifact_repo_root_path, start=run_artifact_root_path
        )
        # If the paths are equal, then use empty string over "./" for ListArtifact compatibility.
        self.run_relative_artifact_repo_root_path = (
            "" if run_artifact_root_path == artifact_repo_root_path else run_relative_root_path
        )
        # Limit the number of threads used for artifact uploads, using at most 8 threads or
        # 2 * the number of CPU cores available on the system (whichever is smaller)
        num_cpus = os.cpu_count() or 4
        num_artifact_workers = min(num_cpus * 2, 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=num_artifact_workers)

    @staticmethod
    def _extract_run_id(artifact_uri):
        """
        The artifact_uri is expected to be
        dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>
        Once the path from the input uri is extracted and normalized, it is
        expected to be of the form
        databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>

        Hence the run_id is the 4th element of the normalized path.

        :return: run_id extracted from the artifact_uri
        """
        artifact_path = extract_and_normalize_path(artifact_uri)
        return artifact_path.split("/")[3]

    def _call_endpoint(self, service, api, json_body):
        db_creds = get_databricks_host_creds(self.databricks_profile_uri)
        endpoint, method = _SERVICE_AND_METHOD_TO_INFO[service][api]
        response_proto = api.Response()
        return call_endpoint(db_creds, endpoint, method, json_body, response_proto)

    def _get_run_artifact_root(self, run_id):
        json_body = message_to_json(GetRun(run_id=run_id))
        run_response = self._call_endpoint(MlflowService, GetRun, json_body)
        return run_response.run.info.artifact_uri

    def _get_write_credential_infos(self, run_id, paths=None):
        write_credential_infos = []
        page_token = None
        while True:
            if page_token:
                json_body = message_to_json(GetCredentialsForWrite(run_id=run_id, path=paths))
            else:
                json_body = message_to_json(GetCredentialsForWrite(run_id=run_id, path=paths, page_token=page_token))

            response = self._call_endpoint(
                DatabricksMlflowArtifactsService, GetCredentialsForWrite, json_body
            )
            write_credential_infos += response.credential_infos
            page_token = response.next_page_token

            if not page_token or len(response.credential_infos) == 0:
                break

        return write_credential_infos

    def _get_read_credential_infos(self, run_id, paths=None):
        read_credential_infos = []
        page_token = None
        while True:
            if page_token:
                json_body = message_to_json(GetCredentialsForRead(run_id=run_id, path=paths))
            else:
                json_body = message_to_json(GetCredentialsForRead(run_id=run_id, path=paths, page_token=page_token))

            response = self._call_endpoint(
                DatabricksMlflowArtifactsService, GetCredentialsForRead, json_body
            )
            read_credential_infos += response.credential_infos
            page_token = response.next_page_token

            if not page_token or len(response.credential_infos) == 0:
                break

        return read_credential_infos

    def _extract_headers_from_credentials(self, headers):
        return {header.name: header.value for header in headers}

    def _azure_upload_file(self, credentials, local_file, artifact_path):
        """
        Uploads a file to a given Azure storage location.
        The function uses a file chunking generator with 100 MB being the size limit for each chunk.
        This limit is imposed by the stage_block API in azure-storage-blob.
        In the case the file size is large and the upload takes longer than the validity of the
        given credentials, a new set of credentials are generated and the operation continues. This
        is the reason for the first nested try-except block
        Finally, since the prevailing credentials could expire in the time between the last
        stage_block and the commit, a second try-except block refreshes credentials if needed.
        """
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            uploading_block_list = list()
            for chunk in yield_file_in_chunks(local_file, _AZURE_MAX_BLOCK_CHUNK_SIZE):
                # Base64-encode a UUID, producing a UTF8-encoded bytestring. Then, decode
                # the bytestring for compliance with Azure Blob Storage API requests
                block_id = base64.b64encode(uuid.uuid4().hex.encode()).decode("utf-8")
                try:
                    put_block(credentials.signed_uri, block_id, chunk, headers=headers)
                except requests.HTTPError as e:
                    if e.response.status_code in [401, 403]:
                        _logger.info(
                            "Failed to authorize request, possibly due to credential expiration."
                            " Refreshing credentials and trying again..."
                        )
                        credentials = self._get_write_credential_infos(
                            run_id=self.run_id, paths=[artifact_path]
                        )[0]
                        put_block(credentials.signed_uri, block_id, chunk, headers=headers)
                    else:
                        raise e
                uploading_block_list.append(block_id)
            try:
                put_block_list(credentials.signed_uri, uploading_block_list, headers=headers)
            except requests.HTTPError as e:
                if e.response.status_code in [401, 403]:
                    _logger.info(
                        "Failed to authorize request, possibly due to credential expiration."
                        " Refreshing credentials and trying again..."
                    )
                    credentials = self._get_write_credential_infos(
                        run_id=self.run_id, paths=[artifact_path]
                    )[0]
                    put_block_list(credentials.signed_uri, uploading_block_list, headers=headers)
                else:
                    raise e
        except Exception as err:
            raise MlflowException(err)

    def _signed_url_upload_file(self, credentials, local_file):
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            signed_write_uri = credentials.signed_uri
            # Putting an empty file in a request by reading file bytes gives 501 error.
            if os.stat(local_file).st_size == 0:
                with rest_utils.cloud_storage_http_request(
                    "put", signed_write_uri, "", headers=headers
                ) as response:
                    response.raise_for_status()
            else:
                with open(local_file, "rb") as file:
                    with rest_utils.cloud_storage_http_request(
                        "put", signed_write_uri, file, headers=headers
                    ) as response:
                        response.raise_for_status()
        except Exception as err:
            raise MlflowException(err)

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, dst_artifact_path):
        if cloud_credential_info.type == ArtifactCredentialType.AZURE_SAS_URI:
            self._azure_upload_file(cloud_credential_info, src_file_path, dst_artifact_path)
        elif cloud_credential_info.type in [
            ArtifactCredentialType.AWS_PRESIGNED_URL,
            ArtifactCredentialType.GCP_SIGNED_URL,
        ]:
            self._signed_url_upload_file(cloud_credential_info, src_file_path)
        else:
            raise MlflowException(
                message="Cloud provider not supported.", error_code=INTERNAL_ERROR
            )

    def _download_from_cloud(self, cloud_credential_info, dst_local_file_path):
        """
        Downloads a file from the input `cloud_credential_info` and saves it to
        `dst_local_file_path`.
        """
        if cloud_credential_info.type not in [
            ArtifactCredentialType.AZURE_SAS_URI,
            ArtifactCredentialType.AWS_PRESIGNED_URL,
            ArtifactCredentialType.GCP_SIGNED_URL,
        ]:
            raise MlflowException(
                message="Cloud provider not supported.", error_code=INTERNAL_ERROR
            )
        try:
            download_file_using_http_uri(cloud_credential_info.signed_read_uri, dst_local_file_path, _DOWNLOAD_CHUNK_SIZE)
        except Exception as err:
            raise MlflowException(err)

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        artifact_path = artifact_path or ""
        artifact_path = posixpath.join(artifact_path, basename)
        if len(artifact_path) > 0:
            run_relative_artifact_path = posixpath.join(
                self.run_relative_artifact_repo_root_path, artifact_path
            )
        else:
            run_relative_artifact_path = self.run_relative_artifact_repo_root_path
        write_credential_info = self._get_write_credential_infos(run_id=self.run_id, paths=[run_relative_artifact_path])[0]
        self._upload_to_cloud(
            cloud_credential_info=write_credential_info,
            src_file_path=local_file,
            dst_artifact_path=run_relative_artifact_path
        )

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Parallelized implementation of `download_artifacts` for Databricks.
        """
        artifact_path = artifact_path or ""

        file_paths_to_upload = []
        for (dirpath, _, filenames) in os.walk(local_dir):
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                file_paths_to_upload.append(file_path)

        write_credential_infos = self._get_write_credential_infos(run_id=self.run_id, paths=file_paths_to_upload)

        inflight_uploads = {}
        for file_path, write_credential_info in zip(file_paths_to_upload, write_credential_infos):
            upload_future = self.thread_pool.submit(
                self._upload_to_cloud,
                cloud_credential_info=write_credential_info,
                src_file_path=file_path,
                dst_artifact_path=artifact_subdir,
            )
            inflight_uploads[file_path] = upload_future

        # Join futures to ensure that all artifacts have been uploaded prior to returning
        failed_uploads = {}
        for (src_file_path, upload_future) in inflight_uploads.items():
            try:
                upload_future.result()
            except Exception as e:
                failed_uploads[src_file_path] = repr(e)

        if len(failed_uploads) > 0:
            raise MlflowException(
                message=(
                    "The following failures occurred while uploading one or more artifacts"
                    " to {artifact_root}: {failures}".format(
                        artifact_root=self.artifact_uri, failures=failed_uploads,
                    )
                )
            )

    def list_artifacts(self, path=None):
        if path:
            run_relative_path = posixpath.join(self.run_relative_artifact_repo_root_path, path)
        else:
            run_relative_path = self.run_relative_artifact_repo_root_path
        infos = []
        page_token = None
        while True:
            if page_token:
                json_body = message_to_json(
                    ListArtifacts(run_id=self.run_id, path=run_relative_path, page_token=page_token)
                )
            else:
                json_body = message_to_json(
                    ListArtifacts(run_id=self.run_id, path=run_relative_path)
                )
            response = self._call_endpoint(MlflowService, ListArtifacts, json_body)
            artifact_list = response.files
            # If `path` is a file, ListArtifacts returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if (
                len(artifact_list) == 1
                and artifact_list[0].path == run_relative_path
                and not artifact_list[0].is_dir
            ):
                return []
            for output_file in artifact_list:
                file_rel_path = posixpath.relpath(
                    path=output_file.path, start=self.run_relative_artifact_repo_root_path
                )
                artifact_size = None if output_file.is_dir else output_file.file_size
                infos.append(FileInfo(file_rel_path, output_file.is_dir, artifact_size))
            if len(artifact_list) == 0 or not response.next_page_token:
                break
            page_token = response.next_page_token
        return infos

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Parallelized implementation of `download_artifacts` for Databricks.
        """

        # Represents an in-progress file artifact download to a local filesystem location
        InflightDownload = namedtuple(
            'InflightDownload',
            [
                # The artifact path, given relative to the repository's artifact root location
                "artifact_src_path",
                # The local filesystem destination path to which artifacts are being downloaded
                "local_dst_path",
                # A future representing the artifact download operation
                "future",
            ],
        )

        def async_download_file_artifacts_from_paths(src_artifact_paths, dst_local_dir_path):
            """
            Initiate an asynchronous download of the file artifacts specified by
            `src_artifact_paths` to the local filesystem directory specified by
            `dst_local_dir_path`.

            :param src_artifact_paths: A list of relative, POSIX-style paths referring to file
                                       artifacts stored within the repository's artifact root
                                       location. Each path should be specified relative to the
                                       artifact repository's artifact root location.
            :param dst_local_dir_path: Absolute path of the local filesystem destination directory
                                       to which to download the specified artifacts. A given
                                       downloaded artifact may be written to a subdirectory of
                                       `dst_local_dir_path` if its source path contains
                                       subdirectories.
            :return: A list of `InflightDownload` objects, each of which represents an inflight
                     asynchronous artifact download. The entry at index `i` corresponds to the
                     artifact path at index `i` of `src_artifact_paths`.
            """
            run_relative_src_artifact_paths = [
                posixpath.join(self.run_relative_artifact_repo_root_path, src_artifact_path)
                for src_artifact_path in src_artifact_paths
            ]
            read_credential_infos = self._get_read_credential_infos(
                run_id=self.run_id, paths=run_relative_src_artifact_paths
            )

            inflight_downloads = []
            for src_artifact_path, read_credential_info in zip(src_artifact_paths, read_credential_infos):
                local_dst_path = self._create_download_destination(
                    src_artifact_path=src_artifact_path, dst_local_dir_path=dst_local_dir_path
                )
                download_future = self.thread_pool.submit(
                    self._download_from_cloud,
                    cloud_credential_info=read_credential_info,
                    dst_local_file_path=local_dst_path,
                )
                inflight_downloads.append(
                    InflightDownload(
                        src_artifact_path=src_artifact_path,
                        local_dst_path=local_dst_path,
                        download_future=download_future,
                    )
                )

            return inflight_downloads

        def async_download_artifact_dir(src_artifact_dir_path, dst_local_dir_path):
            """
            Initiate an asynchronous download of the artifact directory specified by
            `src_artifact_dir_path` to the local filesystem directory specified by
            `dst_local_dir_path`.

            This implementation is adapted from
            https://github.com/mlflow/mlflow/blob/a776b54fa8e1beeca6a984864c6375e9ed38f8c0/mlflow/
            store/artifact/artifact_repo.py#L93.

            :param src_artifact_dir_path: A relative, POSIX-style path referring to a directory of
                                          of artifacts stored within the repository's artifact root
                                          location. `src_artifact_dir_path` should be specified
                                          relative to the repository's artifact root location.
            :param dst_local_dir_path: Absolute path of the local filesystem destination directory
                                       to which to download the specified artifact directory. The
                                       downloaded artifacts may be written to a subdirectory of
                                       `dst_local_dir_path` if `src_artifact_dir_path` contains
                                       subdirectories.
            :return: A tuple whose first element is the destination directory of the downloaded
                     artifacts on the local filesystem and whose second element is a list of
                     `InflightDownload` objects, each of which represents an inflight asynchronous
                     download operation for a file in the specified artifact directory.
            """
            local_dir = os.path.join(dst_local_dir_path, src_artifact_dir_path)
            inflight_downloads = []
            dir_content = [  # prevent infinite loop, sometimes the dir is recursively included
                file_info
                for file_info in self.list_artifacts(src_artifact_dir_path)
                if file_info.path != "." and file_info.path != src_artifact_dir_path
            ]

            if not dir_content:  # empty dir
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)
            else:
                inflight_downloads += async_download_file_artifacts_from_paths(
                    src_artifact_paths=[
                        artifact_info.path for artifact_info in dir_content
                        if not artifact_info.is_dir
                    ],
                    dst_local_dir_path=dst_local_dir_path,
                )

                for dir_artifact_info in [
                    artifact_info for artifact_info in dir_content if artifact_info.is_dir
                ]:
                    inflight_downloads += async_download_artifact_dir(
                        src_artifact_dir_path=dir_artifact_info.path,
                        dst_local_dir_path=dst_local_dir_path,
                    )[1]

            return local_dir, inflight_downloads

        if dst_path is None:
            dst_path = tempfile.mkdtemp()
        dst_path = os.path.abspath(dst_path)

        if not os.path.exists(dst_path):
            raise MlflowException(
                message=(
                    "The destination path for downloaded artifacts does not"
                    " exist! Destination path: {dst_path}".format(dst_path=dst_path)
                ),
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        elif not os.path.isdir(dst_path):
            raise MlflowException(
                message=(
                    "The destination path for downloaded artifacts must be a directory!"
                    " Destination path: {dst_path}".format(dst_path=dst_path)
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        if self._is_directory(artifact_path):
            local_dst_path, inflight_downloads = async_download_artifact_dir(
                src_artifact_dir_path=artifact_path, dst_local_dir_path=dst_path
            )
        else:
            inflight_downloads = async_download_file_artifacts_from_paths(
                src_artifact_paths=[artifact_path], dst_local_dir_path=dst_path
            )
            assert len(inflight_downloads) == 1, "Expected one inflight download for a file artifact, got {} downloads".format(len(inflight_downloads))
            local_dst_path = inflight_downloads[0].local_dst_path

        # Join futures to ensure that all artifacts have been downloaded prior to returning
        failed_downloads = {}
        for inflight_download in inflight_downloads:
            try:
                inflight_download.download_future.result()
            except Exception as e:
                failed_downloads[inflight_download.src_artifact_path] = repr(e)

        if len(failed_downloads) > 0:
            raise MlflowException(
                message=(
                    "The following failures occurred while downloading one or more"
                    " artifacts from {artifact_root}: {failures}".format(
                        artifact_root=self.artifact_uri, failures=failed_downloads,
                    )
                )
            )

        return local_dst_path

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
