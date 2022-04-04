import hashlib
import json
import logging
import os
import shutil

from mlflow.exceptions import ExecutionException
from mlflow.utils import process


# Environment variable indicating a path to a conda installation. MLflow will default to running
# "conda" if unset
MLFLOW_CONDA_HOME = "MLFLOW_CONDA_HOME"
# Environment variable indicated the name of the command that should be used to create environments.
# If it is unset, it will default to "conda". This command must be in the $PATH when the user runs,
# or within MLFLOW_CONDA_HOME if that is set. For example, let's say we want to use mamba
# (https://github.com/mamba-org/mamba) instead of conda to create environments. Then:
# > conda install mamba -n base -c conda-forge
# > MLFLOW_CONDA_CREATE_ENV_CMD="mamba"
# > mlflow run ...
MLFLOW_CONDA_CREATE_ENV_CMD = "MLFLOW_CONDA_CREATE_ENV_CMD"

_logger = logging.getLogger(__name__)


def get_conda_command(conda_env_name):
    #  Checking for newer conda versions
    if os.name != "nt" and ("CONDA_EXE" in os.environ or "MLFLOW_CONDA_HOME" in os.environ):
        conda_path = get_conda_bin_executable("conda")
        activate_conda_env = [
            "source {}/../etc/profile.d/conda.sh".format(os.path.dirname(conda_path))
        ]
        activate_conda_env += ["conda activate {} 1>&2".format(conda_env_name)]
    else:
        activate_path = get_conda_bin_executable("activate")
        # in case os name is not 'nt', we are not running on windows. It introduces
        # bash command otherwise.
        if os.name != "nt":
            return ["source %s %s 1>&2" % (activate_path, conda_env_name)]
        else:
            return ["conda activate %s" % (conda_env_name)]
    return activate_conda_env


def get_conda_bin_executable(executable_name):
    """
    Return path to the specified executable, assumed to be discoverable within the 'bin'
    subdirectory of a conda installation.

    The conda home directory (expected to contain a 'bin' subdirectory) is configurable via the
    ``mlflow.projects.MLFLOW_CONDA_HOME`` environment variable. If
    ``mlflow.projects.MLFLOW_CONDA_HOME`` is unspecified, this method simply returns the passed-in
    executable name.
    """
    conda_home = os.environ.get(MLFLOW_CONDA_HOME)
    if conda_home:
        return os.path.join(conda_home, "bin/%s" % executable_name)
    # Use CONDA_EXE as per https://github.com/conda/conda/issues/7126
    if "CONDA_EXE" in os.environ:
        conda_bin_dir = os.path.dirname(os.environ["CONDA_EXE"])
        return os.path.join(conda_bin_dir, executable_name)
    return executable_name


def _get_conda_env_name(conda_env_path, env_id=None):
    conda_env_contents = open(conda_env_path).read() if conda_env_path else ""
    if env_id:
        conda_env_contents += env_id
    return "mlflow-%s" % hashlib.sha1(conda_env_contents.encode("utf-8")).hexdigest()


def _get_conda_executable_for_create_env():
    """
    Returns the executable that should be used to create environments. This is "conda"
    by default, but it can be set to something else by setting the environment variable

    """
    conda_env_create_cmd = os.environ.get(MLFLOW_CONDA_CREATE_ENV_CMD)
    if conda_env_create_cmd is not None:
        conda_env_create_path = get_conda_bin_executable(conda_env_create_cmd)
    else:
        # Use the same as conda_path
        conda_env_create_path = get_conda_bin_executable("conda")

    return conda_env_create_path


def _list_conda_environments():
    prc = process._exec_cmd([get_conda_bin_executable("conda"), "env", "list", "--json"])
    return list(map(os.path.basename, json.loads(prc.stdout).get("envs", [])))


def _get_conda_env_root_dir_env(conda_env_root_dir):
    if conda_env_root_dir is not None:
        # See https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-environment-directories-envs-dirs
        # and https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs
        return {
            "CONDA_ENVS_PATH": conda_env_root_dir,
            "CONDA_PKGS_DIRS": os.path.join(conda_env_root_dir, "pkgs")
        }
    else:
        return None


def get_or_create_conda_env(conda_env_path, env_id=None, capture_output=False, conda_env_root_dir=None):
    """
    Given a `Project`, creates a conda environment containing the project's dependencies if such a
    conda environment doesn't already exist. Returns the name of the conda environment.
    :param conda_env_path: Path to a conda yaml file.
    :param env_id: Optional string that is added to the contents of the yaml file before
                   calculating the hash. It can be used to distinguish environments that have the
                   same conda dependencies but are supposed to be different based on the context.
                   For example, when serving the model we may install additional dependencies to the
                   environment after the environment has been activated.
    :param capture_output: Specify the capture_output argument while executing the
                           "conda env create" command.
    :param conda_env_root_dir: Root path for conda env. If None, use default one. Note if this is
                                set, conda package cache path becomes "conda_env_root_path/pkgs"
                                instead of the global package cache path.
    """

    conda_path = get_conda_bin_executable("conda")
    conda_env_create_path = _get_conda_executable_for_create_env()

    try:
        process._exec_cmd([conda_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException(
            "Could not find Conda executable at {0}. "
            "Ensure Conda is installed as per the instructions at "
            "https://conda.io/projects/conda/en/latest/"
            "user-guide/install/index.html. "
            "You can also configure MLflow to look for a specific "
            "Conda executable by setting the {1} environment variable "
            "to the path of the Conda executable".format(conda_path, MLFLOW_CONDA_HOME)
        )

    try:
        process._exec_cmd([conda_env_create_path, "--help"], throw_on_error=False)
    except EnvironmentError:
        raise ExecutionException(
            "You have set the env variable {0}, but {1} does not exist or "
            "it is not working properly. Note that {1} and the conda executable need to be "
            "in the same conda environment. You can change the search path by"
            "modifying the env variable {2}".format(
                MLFLOW_CONDA_CREATE_ENV_CMD,
                conda_env_create_path,
                MLFLOW_CONDA_HOME,
            )
        )

    additional_env = _get_conda_env_root_dir_env(conda_env_root_dir)

    project_env_name = _get_conda_env_name(conda_env_path, env_id)
    if conda_env_root_dir is not None:
        # Append a suffix "-isolated" because if a conda env name exist in
        # default conda env root dir, then if we set new "CONDA_ENVS_PATH" and run "conda env create"
        # with the same conda env name, "CondaValueError: prefix already exists" error will
        # be raised.
        project_env_name = project_env_name + "-isolated"

    if project_env_name not in _list_conda_environments():
        _logger.info("=== Creating conda environment %s ===", project_env_name)
        if conda_env_root_dir is not None:
            _logger.info("Use isolated conda environment root directory: %s", conda_env_root_dir)
        try:
            if conda_env_path:
                process._exec_cmd(
                    [
                        conda_env_create_path,
                        "env",
                        "create",
                        "-n",
                        project_env_name,
                        "--file",
                        conda_env_path,
                    ],
                    env=additional_env,
                    capture_output=capture_output,
                )
            else:
                process._exec_cmd(
                    [
                        conda_env_create_path,
                        "create",
                        "--channel",
                        "conda-forge",
                        "--yes",
                        "--override-channels",
                        "-n",
                        project_env_name,
                        "python",
                    ],
                    env=additional_env,
                    capture_output=capture_output,
                )
        except Exception:
            try:
                if project_env_name in _list_conda_environments():
                    _logger.warning(
                        "Encountered unexpected error while creating conda environment. "
                        "Removing %s.",
                        project_env_name,
                    )
                    process._exec_cmd(
                        [
                            conda_path,
                            "remove",
                            "--yes",
                            "--name",
                            project_env_name,
                            "--all",
                        ],
                        capture_output=False,
                    )
            except Exception as e:
                _logger.warning(
                    f"Removing conda env '{project_env_name}' failed (error: {repr(e)})."
                )
            raise
    else:
        if conda_env_root_dir is not None:
            _logger.info(
                "Reuse cached conda environment at %s",
                os.path.join(conda_env_root_dir, project_env_name)
            )

    return project_env_name
