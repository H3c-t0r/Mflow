import docker
import logging
import os
import posixpath
import shutil
import tempfile

from mlflow.exceptions import ExecutionException
import mlflow.tracking as tracking
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.utils import file_utils, process
from mlflow.utils.mlflow_tags import MLFLOW_DOCKER_IMAGE_ID, MLFLOW_DOCKER_IMAGE_URI

# FIXME: also in __init__.py, figure out where to keep
# Environment variable indicating a path to a conda installation. MLflow will default to running
# "conda" if unset
MLFLOW_CONDA_HOME = "MLFLOW_CONDA_HOME"
_GENERATED_DOCKERFILE_NAME = "Dockerfile.mlflow-autogenerated"
_PROJECT_TAR_ARCHIVE_NAME = "mlflow-project-docker-build-context"
_MLFLOW_DOCKER_TRACKING_DIR_PATH = "/mlflow/tmp/mlruns"


_logger = logging.getLogger(__name__)


def _build_docker_image(work_dir, repository_uri, base_image, run_id):
    """
    Build a docker image containing the project in `work_dir`, using the base image.
    """
    image_uri = _get_docker_image_uri(repository_uri=repository_uri, work_dir=work_dir)
    dockerfile = (
        "FROM {imagename}\n"
        "COPY {build_context_path}/ /mlflow/projects/code/\n"
        "WORKDIR /mlflow/projects/code/\n"
    ).format(imagename=base_image, build_context_path=_PROJECT_TAR_ARCHIVE_NAME)
    build_ctx_path = _create_docker_build_ctx(work_dir, dockerfile)
    with open(build_ctx_path, 'rb') as docker_build_ctx:
        _logger.info("=== Building docker image %s ===", image_uri)
        client = docker.from_env()
        image, _ = client.images.build(
            tag=image_uri, forcerm=True,
            dockerfile=posixpath.join(_PROJECT_TAR_ARCHIVE_NAME, _GENERATED_DOCKERFILE_NAME),
            fileobj=docker_build_ctx, custom_context=True, encoding="gzip")
    try:
        os.remove(build_ctx_path)
    except Exception:  # pylint: disable=broad-except
        _logger.info("Temporary docker context file %s was not deleted.", build_ctx_path)
    tracking.MlflowClient().set_tag(run_id,
                                    MLFLOW_DOCKER_IMAGE_URI,
                                    image_uri)
    tracking.MlflowClient().set_tag(run_id,
                                    MLFLOW_DOCKER_IMAGE_ID,
                                    image.id)
    return image


def _create_docker_build_ctx(work_dir, dockerfile_contents):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    directory = tempfile.mkdtemp()
    try:
        dst_path = os.path.join(directory, "mlflow-project-contents")
        shutil.copytree(src=work_dir, dst=dst_path)
        with open(os.path.join(dst_path, _GENERATED_DOCKERFILE_NAME), "w") as handle:
            handle.write(dockerfile_contents)
        _, result_path = tempfile.mkstemp()
        file_utils.make_tarfile(
            output_filename=result_path,
            source_dir=dst_path, archive_name=_PROJECT_TAR_ARCHIVE_NAME)
    finally:
        shutil.rmtree(directory)
    return result_path


def _get_docker_image_uri(repository_uri, work_dir):
    """
    Returns an appropriate Docker image URI for a project based on the git hash of the specified
    working directory.

    :param repository_uri: The URI of the Docker repository with which to tag the image. The
                           repository URI is used as the prefix of the image URI.
    :param work_dir: Path to the working directory in which to search for a git commit hash
    """
    repository_uri = repository_uri if repository_uri else "docker-project"
    # Optionally include first 7 digits of git SHA in tag name, if available.
    git_commit = _get_git_commit(work_dir)
    version_string = ":" + git_commit[:7] if git_commit else ""
    return repository_uri + version_string


def _get_entry_point_command(project, entry_point, parameters, storage_dir):
    """
    Returns the shell command to execute in order to run the specified entry point.
    :param project: Project containing the target entry point
    :param entry_point: Entry point to run
    :param parameters: Parameters (dictionary) for the entry point command
    :param storage_dir: Base local directory to use for downloading remote artifacts passed to
                        arguments of type 'path'. If None, a temporary base directory is used.
    """
    storage_dir_for_run = _get_storage_dir(storage_dir)
    _logger.info(
        "=== Created directory %s for downloading remote URIs passed to arguments of"
        " type 'path' ===",
        storage_dir_for_run)
    commands = []
    commands.append(
        project.get_entry_point(entry_point).compute_command(parameters, storage_dir_for_run))
    return commands


def _get_run_env_vars(run_id, experiment_id):
    """
    Returns a dictionary of environment variable key-value pairs to set in subprocess launched
    to run MLflow projects.
    """
    return {
        tracking._RUN_ID_ENV_VAR: run_id,
        tracking._TRACKING_URI_ENV_VAR: tracking.get_tracking_uri(),
        tracking._EXPERIMENT_ID_ENV_VAR: str(experiment_id),
    }


def _get_storage_dir(storage_dir):
    if storage_dir is not None and not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return tempfile.mkdtemp(dir=storage_dir)


def _validate_docker_env(project):
    if not project.name:
        raise ExecutionException("Project name in MLProject must be specified when using docker "
                                 "for image tagging.")
    if not project.docker_env.get('image'):
        raise ExecutionException("Project with docker environment must specify the docker image "
                                 "to use via an 'image' field under the 'docker_env' field.")


def _validate_docker_installation():
    """
    Verify if Docker is installed on host machine.
    """
    try:
        docker_path = "docker"
        process.exec_cmd([docker_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException("Could not find Docker executable. "
                                 "Ensure Docker is installed as per the instructions "
                                 "at https://docs.docker.com/install/overview/.")
