from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
)


class DatabricksJobRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_job()

    def tags(self):
        job_id = databricks_utils.get_notebook_id()
        job_run_id = databricks_utils.get_job_id()
        webapp_url = databricks_utils.get_webapp_url()
        tags = {
            MLFLOW_SOURCE_NAME: "job/{job_id}/run/{job_run_id}".format(
                job_id=job_id, job_run_id=job_run_id)
                if job_id is not None and job_run_id is not None else None,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB)
        }
        if job_id is not None:
            tags[MLFLOW_DATABRICKS_JOB_ID] = job_id
        if job_run_id is not None:
            tags[MLFLOW_DATABRICKS_JOB_RUN_ID] = job_run_id
        if webapp_url is not None:
            tags[MLFLOW_DATABRICKS_WEBAPP_URL] = webapp_url
        return tags
