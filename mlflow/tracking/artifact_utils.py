"""
Utilities for dealing with artifacts in the context of a Run.
"""
import posixpath
import shutil
import tempfile

from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.uri import append_to_uri_path


def get_artifact_uri(run_id, artifact_path=None):
    """
    Get the absolute URI of the specified artifact in the specified run. If `path` is not specified,
    the artifact root URI of the specified run will be returned; calls to ``log_artifact``
    and ``log_artifacts`` write artifact(s) to subdirectories of the artifact root URI.

    :param run_id: The ID of the run for which to obtain an absolute artifact URI.
    :param artifact_path: The run-relative artifact path. For example,
                          ``path/to/artifact``. If unspecified, the artifact root URI for the
                          specified run will be returned.
    :return: An *absolute* URI referring to the specified artifact or the specified run's artifact
             root. For example, if an artifact path is provided and the specified run uses an
             S3-backed  store, this may be a uri of the form
             ``s3://<bucket_name>/path/to/artifact/root/path/to/artifact``. If an artifact path
             is not provided and the specified run uses an S3-backed store, this may be a URI of
             the form ``s3://<bucket_name>/path/to/artifact/root``.
    """
    if not run_id:
        raise MlflowException(
            message="A run_id must be specified in order to obtain an artifact uri!",
            error_code=INVALID_PARAMETER_VALUE)

    store = _get_store()
    run = store.get_run(run_id)
    # Maybe move this method to RunsArtifactRepository so the circular dependency is clearer.
    assert urllib.parse.urlparse(run.info.artifact_uri).scheme != "runs"  # avoid an infinite loop
    if artifact_path is None:
        return run.info.artifact_uri
    else:
        return append_to_uri_path(run.info.artifact_uri, artifact_path)


# TODO: This would be much simpler if artifact_repo.download_artifacts could take the absolute path
# or no path.
def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    :param artifact_uri: The *absolute* URI of the artifact to download.
    :param output_path: The local filesystem path to which to download the artifact. If unspecified,
                        a local output path will be created.
    """
    parsed_uri = urllib.parse.urlparse(artifact_uri)
    prefix = ""
    if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
        # relative path is a special case, urllib does not reconstruct it properly
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    # For models:/ URIs, it doesn't make sense to initialize a ModelsArtifactRepository with only
    # the model name portion of the URI, then call download_artifacts with the version info.
    if ModelsArtifactRepository.is_models_uri(artifact_uri):
        root_uri = artifact_uri
        artifact_path = ""
    else:
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)

    return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
        artifact_path=artifact_path, dst_path=output_path)


def _upload_artifacts_to_databricks(source, run_id, databricks_profile_uri=None):
    """
    Copy the artifacts from ``source`` to the destination Databricks workspace (DBFS) given by
    ``databricks_profile_uri`` or the current tracking URI.
    :param source: Source location for the artifacts to copy.
    :param run_id: Run ID to associate the artifacts with.
    :param databricks_profile_uri: Specifies the destination Databricks host. If not given,
        defaults to the current tracking URI.
    :return: The DBFS location in the target Databricks workspace the model files have been
        uploaded to.
    """
    import uuid
    local_dir = tempfile.mkdtemp()
    try:
        _download_artifact_from_uri(source, local_dir)
        dest_root = 'dbfs:/databricks/mlflow/tmp-external-source/'  # TODO: "/" or "-"?
        dest_repo = DbfsRestArtifactRepository(dest_root, databricks_profile_uri)
        dest_dir = run_id if run_id else uuid.uuid1()
        dest_repo.log_artifacts(local_dir, artifact_path=dest_dir)
        return dest_root + dest_dir  # new source
    finally:
        shutil.rmtree(local_dir)
    # NOTE: we can't easily delete the target temp location due to the async nature
    # of the model version creation.
