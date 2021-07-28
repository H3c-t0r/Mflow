"""
This module provides a set of utilities for interpreting and creating requirements files
(e.g. pip's `requirements.txt`), which is useful for managing ML software environments.
"""

import sys
import subprocess
import tempfile
import os
import pkg_resources
import importlib_metadata
from itertools import filterfalse, chain
from collections import namedtuple

from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils.versioning import _strip_dev_version_suffix
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils import _capture_modules


def _is_comment(line):
    return line.startswith("#")


def _is_empty(line):
    return line == ""


def _strip_inline_comment(line):
    return line[: line.find(" #")].rstrip() if " #" in line else line


def _is_requirements_file(line):
    return line.startswith("-r ") or line.startswith("--requirement ")


def _is_constraints_file(line):
    return line.startswith("-c ") or line.startswith("--constraint ")


def _join_continued_lines(lines):
    """
    Joins lines ending with '\\'.

    >>> _join_continued_lines["a\\", "b\\", "c"]
    >>> 'abc'
    """
    continued_lines = []

    for line in lines:
        if line.endswith("\\"):
            continued_lines.append(line.rstrip("\\"))
        else:
            continued_lines.append(line)
            yield "".join(continued_lines)
            continued_lines.clear()

    # The last line ends with '\'
    if continued_lines:
        yield "".join(continued_lines)


# Represents a pip requirement.
#
# :param req_str: A requirement string (e.g. "scikit-learn == 0.24.2").
# :param is_constraint: A boolean indicating whether this requirement is a constraint.
_Requirement = namedtuple("_Requirement", ["req_str", "is_constraint"])


def _parse_requirements(requirements_file, is_constraint):
    """
    A simplified version of `pip._internal.req.parse_requirements` which performs the following
    operations on the given requirements file and yields the parsed requirements.

    - Remove comments and blank lines
    - Join continued lines
    - Resolve requirements file references (e.g. '-r requirements.txt')
    - Resolve constraints file references (e.g. '-c constraints.txt')

    :param requirements_file: A string path to a requirements file on the local filesystem.
    :param is_constraint: Indicates the parsed requirements file is a constraint file.
    :return: A list of ``_Requirement`` instances.

    References:
    - `pip._internal.req.parse_requirements`:
      https://github.com/pypa/pip/blob/7a77484a492c8f1e1f5ef24eaf71a43df9ea47eb/src/pip/_internal/req/req_file.py#L118
    - Requirements File Format:
      https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format
    - Constraints Files:
      https://pip.pypa.io/en/stable/user_guide/#constraints-files
    """
    with open(requirements_file) as f:
        lines = f.read().splitlines()

    lines = map(str.strip, lines)
    lines = map(_strip_inline_comment, lines)
    lines = _join_continued_lines(lines)
    lines = filterfalse(_is_comment, lines)
    lines = filterfalse(_is_empty, lines)

    for line in lines:
        if _is_requirements_file(line):
            req_file = line.split(maxsplit=1)[1]
            # If `req_file` is an absolute path, `os.path.join` returns `req_file`:
            # https://docs.python.org/3/library/os.path.html#os.path.join
            abs_path = os.path.join(os.path.dirname(requirements_file), req_file)
            yield from _parse_requirements(abs_path, is_constraint=False)
        elif _is_constraints_file(line):
            req_file = line.split(maxsplit=1)[1]
            abs_path = os.path.join(os.path.dirname(requirements_file), req_file)
            yield from _parse_requirements(abs_path, is_constraint=True)
        else:
            yield _Requirement(line, is_constraint)


def _flatten(iterable):
    return chain.from_iterable(iterable)


def _canonicalize_package_name(pkg_name):
    return pkg_name.lower().replace("_", "-")


_MODULE_TO_PACKAGES = importlib_metadata.packages_distributions()


def _module_to_packages(module_name):
    """
    Returns a list of packages that provide the specified module.
    """
    return _MODULE_TO_PACKAGES.get(module_name, [])


def _get_requires_recursive(pkg_name):
    """
    Recursively yields both direct and transitive dependencies of the specified package.
    """
    if pkg_name not in pkg_resources.working_set.by_key:
        return

    package = pkg_resources.working_set.by_key[pkg_name]
    reqs = package.requires()
    if len(reqs) == 0:
        return

    for req in reqs:
        yield req.name
        yield from _get_requires_recursive(req.name)


def _prune_packages(packages):
    """
    Prunes packages required by other packages. For example, `["scikit-learn", "numpy"]` is pruned
    to `["scikit-learn"]`.
    """
    requires = _flatten(map(_get_requires_recursive, packages))
    return set(packages) - set(requires)


def _run_command(cmd):
    """
    Runs the specified command.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    if proc.returncode != 0:
        msg = "\n".join(
            [
                f"Encountered an unexpected error while running {cmd}",
                f"exit status: {proc.returncode}",
                f"stdout: {stdout}",
                f"stderr: {stderr}",
            ]
        )
        raise MlflowException(msg)


def _get_package_version(package):
    """
    Returns the version of the specified packages.
    """
    version = importlib_metadata.version(package)
    # Strips a dev version suffix (e.g. '.dev0') for pyspark in Databricks.
    if package == "pyspark" and is_in_databricks_runtime():
        version = _strip_dev_version_suffix(version)
    return version


def _infer_requirements(model_uri, flavor):
    """
    Infers the pip requirements of the specified model.
    """
    local_model_path = _download_artifact_from_uri(model_uri)

    # Run `_capture_modules.py` to capture modules imported during the loading procedure
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "output.txt")
        _run_command(
            [
                sys.executable,
                _capture_modules.__file__,
                "--model-path",
                local_model_path,
                "--flavor",
                flavor,
                "--output-file",
                output_file,
            ]
        )
        with open(output_file) as f:
            modules = f.read().splitlines()

    packages = _flatten(map(_module_to_packages, modules))
    packages = map(_canonicalize_package_name, packages)
    packages = set(packages) - set(["setuptools", "pkg_resources"])
    packages = _prune_packages(packages)
    return ["{}=={}".format(p, _get_package_version(p)) for p in sorted(packages)]
