import logging
import os
import re
import subprocess

from mlflow.models import FlavorBackend
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)


class RFuncBackend(FlavorBackend):
    """
    Flavor backend implementation for the generic R models.
    Predict and serve locally models with 'crate' flavor.
    """
    version_pattern = re.compile("version ([0-9]+[.][0-9]+[.][0-9]+)")

    def predict(self, model_uri, input_path, output_path, content_type, json_format):
        """
        Generate predictions using R model saved with MLflow.
        Return the prediction results as a JSON.
        """
        with TempDir() as tmp:
            model_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            str_cmd = "mlflow:::mlflow_rfunc_predict('{0}', {1}, {2}, {3})"
            command = str_cmd.format(model_path, _str_optional(input_path),
                                     _str_optional(output_path),
                                     _str_optional(content_type))
            _execute(command)

    def serve(self, model_uri, port, **kwargs):
        """
        Generate R model locally.
        """
        with TempDir() as tmp:
            model_path = _download_artifact_from_uri(model_uri, output_path=tmp.path())
            command = "mlflow::mlflow_rfunc_serve('{0}', port = {1})".format(model_path, port)
            _execute(command)

    def can_score_model(self, **kwargs):
        process = subprocess.Popen(["Rscript", "--version"], close_fds=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.wait() != 0:
            return False

        version = self.version_pattern.search(stderr.decode("utf-8"))
        if not version:
            return False
        version = [int(x) for x in version.group(1).split(".")]
        return version[0] > 3 or version[0] == 3 and version[1] >= 3


def _execute(command):
    env = os.environ.copy()
    import sys
    process = subprocess.Popen(["Rscript", "-e", command], env=env, close_fds=False,
                               stdin=sys.stdin,
                               stdout=sys.stdout,
                               stderr=sys.stderr)
    if process.wait() != 0:
        raise Exception("Command returned non zero exit code.")


def _str_optional(s):
    return "NULL" if s is None else "'{}'".format(str(s))
