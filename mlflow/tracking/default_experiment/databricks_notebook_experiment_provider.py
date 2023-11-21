import logging

from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import (
    MLFLOW_EXPERIMENT_SOURCE_ID,
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
)

_logger = logging.getLogger(__name__)


class DatabricksNotebookExperimentProvider(DefaultExperimentProvider):
    _resolved_notebook_experiment_id = None

    def in_context(self):
        return databricks_utils.is_in_databricks_notebook()

    def get_experiment_id(self):
        _logger.debug("get_experiment_id for DatabricksNotebookExperimentProvider")
        print("get_experiment_id for DatabricksNotebookExperimentProvider")
        if DatabricksNotebookExperimentProvider._resolved_notebook_experiment_id:
            return DatabricksNotebookExperimentProvider._resolved_notebook_experiment_id

        source_notebook_id = databricks_utils.get_notebook_id()
        source_notebook_name = databricks_utils.get_notebook_path()
        tags = {
            MLFLOW_EXPERIMENT_SOURCE_ID: source_notebook_id,
        }
        print(f"source_notebook_id {source_notebook_id}")
        print(f"source_notebook_name {source_notebook_name}")

        # With the presence of the source id, the following is a get or create in which it will
        # return the corresponding experiment if one exists for the repo notebook.
        # For non-repo notebooks, it will raise an exception and we will use source_notebook_id
        try:
            experiment_id = MlflowClient().create_experiment(source_notebook_name, None, tags)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.INVALID_PARAMETER_VALUE
            ):
                print("it was not a repo notebook")
                # If determined that it is not a repo noetbook
                experiment_id = source_notebook_id
            else:
                raise e

        DatabricksNotebookExperimentProvider._resolved_notebook_experiment_id = experiment_id
        _logger.debug(f"experiment_id = {experiment_id}")
        print(f"experiment_id = {experiment_id}")

        return experiment_id


class DatabricksRepoNotebookExperimentProvider(DefaultExperimentProvider):
    _resolved_repo_notebook_experiment_id = None

    def in_context(self):
        return databricks_utils.is_in_databricks_repo_notebook()

    def get_experiment_id(self):
        _logger.debug("get_experiment_id for DatabricksREPONotebookExperimentProvider")
        print("get_experiment_id for DatabricksREPONotebookExperimentProvider")
        if DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id:
            return DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id

        source_notebook_id = databricks_utils.get_notebook_id()
        source_notebook_name = databricks_utils.get_notebook_path()
        tags = {
            MLFLOW_EXPERIMENT_SOURCE_TYPE: "REPO_NOTEBOOK",
            MLFLOW_EXPERIMENT_SOURCE_ID: source_notebook_id,
        }

        # With the presence of the above tags, the following is a get or create in which it will
        # return the corresponding experiment if one exists for the repo notebook.
        # If no corresponding experiment exist, it will create a new one and return
        # the newly created experiment ID.
        try:
            experiment_id = MlflowClient().create_experiment(source_notebook_name, None, tags)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.INVALID_PARAMETER_VALUE
            ):
                # If repo notebook experiment creation isn't enabled, fall back to
                # using the notebook ID
                experiment_id = source_notebook_id
            else:
                raise e

        DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id = (
            experiment_id
        )
        _logger.debug(f"experiment_id = {experiment_id}")
        print(f"experiment_id = {experiment_id}")
        return experiment_id
