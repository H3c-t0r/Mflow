import os
import shutil
import tempfile

import docker

import posixpath
from pathlib import Path

import mlflow.tracking
from mlflow import tracking as tracking
from mlflow.exceptions import ExecutionException
from mlflow.projects import _PROJECT_TAR_ARCHIVE_NAME, _logger, _GENERATED_DOCKERFILE_NAME, _get_run_env_vars, \
    _get_local_uri_or_none, _MLFLOW_DOCKER_TRACKING_DIR_PATH
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.utils import file_utils, process, databricks_utils
from mlflow.utils.mlflow_tags import MLFLOW_DOCKER_IMAGE_URI, MLFLOW_DOCKER_IMAGE_ID
from mlflow.store.artifact_repository_registry import get_artifact_repository

from mlflow.store.local_artifact_repo import LocalArtifactRepository
from mlflow.store.s3_artifact_repo import S3ArtifactRepository


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


def _validate_docker_env(project):
    if not project.name:
        raise ExecutionException("Project name in MLProject must be specified when using docker "
                                 "for image tagging.")
    if not project.docker_env.get('image'):
        raise ExecutionException("Project with docker environment must specify the docker image "
                                 "to use via an 'image' field under the 'docker_env' field.")


def _get_local_artifact_cmd_and_envs(artifact_repo):
    return ["-v", "%s:%s" % (artifact_repo.artifact_uri, artifact_repo.artifact_uri)], {}


def _get_s3_artifact_cmd_and_envs(artifact_repo):
    aws_path = Path.home().joinpath(".aws")
    home_path = Path.home()

    print(home_path)

    volumes = []
    if aws_path.exists():
        volumes = ["-v", "%s:%s" % (str(aws_path), "/.aws")]
    envs = {
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    }
    envs = dict((k, v) for k, v in envs.items() if v is not None)
    return volumes, envs


_artifact_storages = {
    LocalArtifactRepository: _get_local_artifact_cmd_and_envs,
    S3ArtifactRepository: _get_s3_artifact_cmd_and_envs,
}


def _get_docker_artifact_storage_cmd_and_envs(artifact_uri):
    artifact_repo = get_artifact_repository(artifact_uri)
    _get_cmd_and_envs = _artifact_storages.get(type(artifact_repo))
    if _get_cmd_and_envs is not None:
        return _get_cmd_and_envs(artifact_repo)
    else:
        return [], {}


def _get_docker_command(image, active_run):
    docker_path = "docker"
    cmd = [docker_path, "run", "--rm"]
    env_vars = _get_run_env_vars(run_id=active_run.info.run_id,
                                 experiment_id=active_run.info.experiment_id)
    tracking_uri = tracking.get_tracking_uri()
    local_path, container_tracking_uri = _get_local_uri_or_none(tracking_uri)
    artifact_cmds, artifact_envs = \
        _get_docker_artifact_storage_cmd_and_envs(active_run.info.artifact_uri)
    cmd += artifact_cmds
    env_vars.update(artifact_envs)
    if local_path is not None:
        cmd += ["-v", "%s:%s" % (local_path, _MLFLOW_DOCKER_TRACKING_DIR_PATH)]
        env_vars[tracking._TRACKING_URI_ENV_VAR] = container_tracking_uri
    if tracking.utils._is_databricks_uri(tracking_uri):
        db_profile = mlflow.tracking.utils.get_db_profile_from_uri(tracking_uri)
        config = databricks_utils.get_databricks_host_creds(db_profile)
        # We set these via environment variables so that only the current profile is exposed, rather
        # than all profiles in ~/.databrickscfg; maybe better would be to mount the necessary
        # part of ~/.databrickscfg into the container
        env_vars[tracking._TRACKING_URI_ENV_VAR] = 'databricks'
        env_vars['DATABRICKS_HOST'] = config.host
        if config.username:
            env_vars['DATABRICKS_USERNAME'] = config.username
        if config.password:
            env_vars['DATABRICKS_PASSWORD'] = config.password
        if config.token:
            env_vars['DATABRICKS_TOKEN'] = config.token
        if config.ignore_tls_verification:
            env_vars['DATABRICKS_INSECURE'] = config.ignore_tls_verification

    for key, value in env_vars.items():
        cmd += ["-e", "{key}={value}".format(key=key, value=value)]
    cmd += [image.tags[0]]
    return cmd