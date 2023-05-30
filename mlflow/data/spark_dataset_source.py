from typing import TypeVar, Any, Optional, Dict

from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


SparkDatasetSourceType = TypeVar("SparkDatasetSourceType", bound="SparkDatasetSource")


@experimental
class SparkDatasetSource(DatasetSource):
    """
    Represents the source of a dataset stored in a spark table.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        table_name: Optional[str] = None,
        sql: Optional[str] = None,
    ):
        if (path, table_name, sql).count(None) != 2:
            raise MlflowException(
                'Must specify exactly one of "path", "table_name", or "sql"',
                INVALID_PARAMETER_VALUE,
            )
        self._path = path
        self._table_name = table_name
        self._sql = sql

    @staticmethod
    def _get_source_type() -> str:
        return "spark"

    def load(self, **kwargs):
        """
        Loads the dataset source as a Spark Dataset Source.
        :return: An instance of :py:class:`pyspark.sql.DataFrame`.
        """
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        if self._path:
            return spark.read.parquet(self._path)
        if self._table_name:
            return spark.read.table(self._table_name)
        if self._sql:
            return spark.sql(self._sql)

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> SparkDatasetSourceType:
        raise NotImplementedError

    def _to_dict(self) -> Dict[Any, Any]:
        info = {}
        if self._path is not None:
            info["path"] = self._path
        elif self._table_name is not None:
            info["table_name"] = self._table_name
        elif self._sql is not None:
            info["sql"] = self._sql
        return info

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> SparkDatasetSourceType:
        return cls(
            path=source_dict.get("path"),
            table_name=source_dict.get("table_name"),
            sql=source_dict.get("sql"),
        )
