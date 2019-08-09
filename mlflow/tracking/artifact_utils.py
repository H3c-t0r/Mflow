"""
Utilities for dealing with artifacts in the context of a Run.
"""
import posixpath

from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact_repository_registry import get_artifact_repository, \
    get_artifact_repository_type
from mlflow.tracking.utils import _get_store
from mlflow.utils.validation import _validate_db_type_string
from mlflow.store.db_artifact_repo import extract_db_uri_and_path
_INVALID_DB_URI_MSG = "Please refer to https://mlflow.org/docs/latest/tracking.html for " \
                      "format specifications."


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
        return posixpath.join(run.info.artifact_uri, artifact_path)


# TODO: This method does not require a Run and its internals should be moved to
#  data.download_uri (requires confirming that Projects will not break with this change).
# Also this would be much simpler if artifact_repo.download_artifacts could take the absolute path
# or no path.
def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    :param artifact_uri: The *absolute* URI of the artifact to download.
    :param output_path: The local filesystem path to which to download the artifact. If unspecified,
                        a local output path will be created.
    """
    artifact_repo_type = get_artifact_repository_type(artifact_uri)
    if artifact_repo_type == 'filesystem':
        parsed_uri = urllib.parse.urlparse(artifact_uri)
        prefix = ""
        if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
            # relative path is a special case, urllib does not reconstruct it properly
            prefix = parsed_uri.scheme + ":"
            parsed_uri = parsed_uri._replace(scheme="")
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)

        return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
            artifact_path=artifact_path, dst_path=output_path)
    else:
        db_uri, artifact_path = extract_db_uri_and_artifact_path_from_uri(artifact_uri)
        print(db_uri, artifact_path)
        return get_artifact_repository(artifact_uri=db_uri).download_artifacts(
            artifact_path=artifact_path, dst_path=output_path)


def extract_db_uri_and_artifact_path_from_uri(artifact_uri):
    """
    Parse the specified artifact URI to extract the artifact path from the DB_URI.
    The DB_URIs are of the form:
    <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>?<query>.
    Confirm the database type is supported. If a driver is specified,
    confirm it passes a plausible regex.
    """

    parsed_uri = urllib.parse.urlparse(artifact_uri)
    scheme = parsed_uri.scheme
    scheme_plus_count = scheme.count('+')

    if scheme_plus_count == 0:
        db_type = scheme
    elif scheme_plus_count == 1:
        db_type, _ = scheme.split('+')
    else:
        error_msg = "Invalid database scheme in the URI: '%s'. %s" % (scheme, _INVALID_DB_URI_MSG)
        raise MlflowException(error_msg, INVALID_PARAMETER_VALUE)

    _validate_db_type_string(db_type)

    return extract_db_uri_and_path(artifact_uri)



