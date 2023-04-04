import hashlib
import json
from typing import List, Optional, Any, Dict, Union

import numpy as np
import pandas as pd
from functools import cached_property

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class NumpyDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a NumPy dataset for use with MLflow Tracking.
    """

    def __init__(
        self,
        features: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        source: FileSystemDatasetSource,
        targets: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        """
        :param features: A numpy array or list/dict of arrays containing dataset features.
        :param source: The source of the numpy dataset.
        :param targets: TA numpy array or list/dict of arrays containing dataset targets. Optional
        :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                     automatically generated.
        :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                       is automatically computed.
        """
        self._features = features
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        MAX_ROWS = 10000

        flattened_features = self._features.flatten()
        trimmed_features = flattened_features[0:MAX_ROWS]

        md5 = hashlib.md5()

        # hash trimmed feature contents
        try:
            md5.update(pd.util.hash_array(trimmed_features))
        except TypeError:
            md5.update(np.int64(trimmed_features.size))
        # hash full feature dimensions
        for x in self._features.shape:
            md5.update(np.int64(x))

        # hash trimmed targets contents
        if self._targets is not None:
            flattened_targets = self._targets.flatten()
            trimmed_targets = flattened_targets[0:MAX_ROWS]
            try:
                md5.update(pd.util.hash_array(trimmed_targets))
            except TypeError:
                md5.update(np.int64(trimmed_targets.size))
            # hash full feature dimensions
            for x in self._targets.shape:
                md5.update(np.int64(x))

        return md5.hexdigest()[:8]

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """
        base_dict.update(
            {
                "schema": json.dumps({"mlflow_tensorspec": self.schema.to_dict()}),
                "profile": json.dumps(self.profile),
            }
        )
        return base_dict

    @property
    def source(self) -> FileSystemDatasetSource:
        return self._source

    @property
    def features(self) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        return self._features

    @property
    def targets(self) -> Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        return self._targets

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        return {
            "shape": self._features.shape,
        }

    @cached_property
    def schema(self) -> Schema:
        """
        An MLflow TensorSpec schema representing the tensor dataset
        """
        return _infer_schema(self._features)

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the dataset to a collection of pyfunc inputs and outputs for model
        evaluation. Required for use with mlflow.evaluate().
        May not be implemented by all datasets.
        """
        return PyFuncInputsOutputs(self._features, self._targets)


def from_numpy(
    features: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
    source: Union[str, DatasetSource],
    targets: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> NumpyDataset:
    """
    :param features: NumPy features, represented as an np.ndarray, list of np.ndarrays
                    or dictionary of named np.ndarrays.
    :param source: The source from which the NumPy data was derived, e.g. a filesystem
                    path, an S3 URI, an HTTPS URL etc. Attempting to use other source
                     types will throw.
    :param targets: Optional NumPy targets, represented as an np.ndarray, list of
                    np.ndarrays or dictionary of named np.ndarrays.
    :param name: The name of the dataset. If unspecified, a name is generated.
    :param digest: A dataset digest (hash). If unspecified, a digest is computed
                    automatically.
    """
    from mlflow.data.dataset_source_registry import resolve_dataset_source

    resolved_source: FileSystemDatasetSource = resolve_dataset_source(source)
    return NumpyDataset(
        features=features, source=resolved_source, targets=targets, name=name, digest=digest
    )
