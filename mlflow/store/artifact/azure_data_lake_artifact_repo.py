import os
import posixpath
import re
import urllib.parse

import re
from azure.storage.filedatalake import DataLakeServiceClient, DataLakeDirectoryClient
from mlflow.utils.databricks_utils import get_databricks_host_creds

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository

from mlflow.environment_variables import MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT


def _parse_abfss_uri(uri):
    """
    Parse an ABFSS URI in the format "abfs[s]://file_system@account_name.api_url_suffix/path",
    returning a tuple consisting of the filesystem, account name, API URL suffix, and path
    :param uri: ABFSS URI to parse
    :return: A tuple containing the name of the filesystem, account name, API URL suffix, and path
    """
    # https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-abfs-driver#uri-scheme-to-reference-data
    # Format is abfs[s]://file_system@account_name.dfs.core.windows.net/<path>/<path>/<file_name>
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "abfss":
        raise MlflowException("Not an ABFSS URI: %s" % uri)

    match = re.match(
        r"([^@]+)@([^.]+)\.(dfs\.core\.(windows\.net|chinacloudapi\.cn))", parsed.netloc
    )

    if match is None:
        raise MlflowException(
            "ABFSS URI must be of the form "
            "abfss://<filesystem>@<account>.dfs.core.windows.net"
            " or abfss://<filesystem>@<account>.dfs.core.chinacloudapi.cn"
        )
    filesystem = match.group(1)
    account_name = match.group(2)
    api_uri_suffix = match.group(3)
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]
    return filesystem, account_name, api_uri_suffix, path

class AzureDataLakeArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Azure Data Lake Storage Gen2.

    This repository is used with URIs of the form
    ``abfs[s]://file_system@account_name.dfs.core.windows.net/<path>/<path>``.

    :param credential: Azure credential (see options in https://learn.microsoft.com/en-us/python/api/azure-core/azure.core.credentials?view=azure-python)
                       to use to authenticate to storage
    """

    def __init__(self, artifact_uri, credential):
        super().__init__(artifact_uri)

        _DEFAULT_TIMEOUT = 600  # 10 minutes
        self.write_timeout = MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.get() or _DEFAULT_TIMEOUT

        from azure.storage.blob import BlobServiceClient

        (filesystem, account_name, api_uri_suffix, path) = _parse_abfss_uri(artifact_uri)

        account_url = f"https://{account}.{api_uri_suffix}"
        self.client = BlobServiceClient(
            account_url=account_url, credential=credential
        )

    @staticmethod
    def parse_wasbs_uri(uri):
        """Parse a wasbs:// URI, returning (container, storage_account, path, api_uri_suffix)."""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "wasbs":
            raise Exception("Not a WASBS URI: %s" % uri)

        match = re.match(
            r"([^@]+)@([^.]+)\.(blob\.core\.(windows\.net|chinacloudapi\.cn))", parsed.netloc
        )

        if match is None:
            raise Exception(
                "WASBS URI must be of the form "
                "<container>@<account>.blob.core.windows.net"
                " or <container>@<account>.blob.core.chinacloudapi.cn"
            )
        container = match.group(1)
        storage_account = match.group(2)
        api_uri_suffix = match.group(3)
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return container, storage_account, path, api_uri_suffix

    def log_artifact(self, local_file, artifact_path=None):
        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        with open(local_file, "rb") as file:
            container_client.upload_blob(
                dest_path, file, overwrite=True, timeout=self.write_timeout
            )

    def log_artifacts(self, local_dir, artifact_path=None):
        (container, _, dest_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                remote_file_path = posixpath.join(upload_path, f)
                local_file_path = os.path.join(root, f)
                with open(local_file_path, "rb") as file:
                    container_client.upload_blob(
                        remote_file_path, file, overwrite=True, timeout=self.write_timeout
                    )

    def list_artifacts(self, path=None):
        # Newer versions of `azure-storage-blob` (>= 12.4.0) provide a public
        # `azure.storage.blob.BlobPrefix` object to signify that a blob is a directory,
        # while older versions only expose this API internally as
        # `azure.storage.blob._models.BlobPrefix`
        try:
            from azure.storage.blob import BlobPrefix
        except ImportError:
            from azure.storage.blob._models import BlobPrefix

        def is_dir(result):
            return isinstance(result, BlobPrefix)

        (container, _, artifact_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"
        results = container_client.walk_blobs(name_starts_with=prefix)

        for result in results:
            if (
                dest_path == result.name
            ):  # result isn't actually a child of the path we're interested in, so skip it
                continue

            if not result.name.startswith(artifact_path):
                raise MlflowException(
                    "The name of the listed Azure blob does not begin with the specified"
                    " artifact path. Artifact path: {artifact_path}. Blob name:"
                    " {blob_name}".format(artifact_path=artifact_path, blob_name=result.name)
                )

            if is_dir(result):
                subdir = posixpath.relpath(path=result.name, start=artifact_path)
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                infos.append(FileInfo(subdir, is_dir=True, file_size=None))
            else:  # Just a plain old blob
                file_name = posixpath.relpath(path=result.name, start=artifact_path)
                infos.append(FileInfo(file_name, is_dir=False, file_size=result.size))

        # The list_artifacts API expects us to return an empty list if the
        # the path references a single file.
        rel_path = dest_path[len(artifact_path) + 1 :]
        if (len(infos) == 1) and not infos[0].is_dir and (infos[0].path == rel_path):
            return []
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        (container, _, remote_root_path, _) = self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        remote_full_path = posixpath.join(remote_root_path, remote_file_path)
        with open(local_path, "wb") as file:
            container_client.download_blob(remote_full_path).readinto(file)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
