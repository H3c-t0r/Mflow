"""
The :py:mod:`mlflow.models.signature` module provides an API for specification of model signature.

Model signature defines schema of model input and output. See :py:class:`mlflow.types.schema.Schema`
for more details on Schema and data types.
"""
import re
import inspect
import logging
from typing import List, Dict, Any, Union, get_type_hints, TYPE_CHECKING


import pandas as pd
import numpy as np

from mlflow.exceptions import MlflowException
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema, _infer_schema_from_type_hint


# At runtime, we don't need  `pyspark.sql.dataframe`
if TYPE_CHECKING:
    try:
        import pyspark.sql.dataframe

        MlflowInferableDataset = Union[
            pd.DataFrame, np.ndarray, Dict[str, np.ndarray], pyspark.sql.dataframe.DataFrame
        ]
    except ImportError:
        MlflowInferableDataset = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]

_logger = logging.getLogger(__name__)


class ModelSignature:
    """
    ModelSignature specifies schema of model's inputs and outputs.

    ModelSignature can be :py:func:`inferred <mlflow.models.infer_signature>` from training dataset
    and model predictions using or constructed by hand by passing an input and output
    :py:class:`Schema <mlflow.types.Schema>`.
    """

    def __init__(self, inputs: Schema, outputs: Schema = None):
        if not isinstance(inputs, Schema):
            raise TypeError(
                "inputs must be mlflow.models.signature.Schema, got '{}'".format(type(inputs))
            )
        if outputs is not None and not isinstance(outputs, Schema):
            raise TypeError(
                "outputs must be either None or mlflow.models.signature.Schema, "
                "got '{}'".format(type(inputs))
            )
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in an MLmodel yaml file.

        :return: dictionary representation with input and output schema represented as json strings.
        """

        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None,
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.

        :param signature_dict: Dictionary representation of model signature.
                               Expected dictionary format:
                               `{'inputs': <json string>, 'outputs': <json string>" }`

        :return: ModelSignature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(signature_dict["inputs"])
        if "outputs" in signature_dict and signature_dict["outputs"] is not None:
            outputs = Schema.from_json(signature_dict["outputs"])
            return cls(inputs, outputs)
        else:
            return cls(inputs)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ModelSignature)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __repr__(self) -> str:
        return (
            "inputs: \n"
            "  {}\n"
            "outputs: \n"
            "  {}\n".format(repr(self.inputs), repr(self.outputs))
        )


def infer_signature(
    model_input: Any, model_output: "MlflowInferableDataset" = None
) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input) and model predictions (output).

    The signature represents model input and output as data frames with (optionally) named columns
    and data type specified as one of types defined in :py:class:`mlflow.types.DataType`.
    This method will raise an exception if the user data contains incompatible types or is not
    passed in one of the supported formats listed below.

    The input should be one of these:
      - pandas.DataFrame
      - pandas.Series
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame
      - scipy.sparse.csr_matrix
      - scipy.sparse.csc_matrix

    The element types should be mappable to one of :py:class:`mlflow.types.DataType`.

    For pyspark.sql.DataFrame inputs, columns of type DateType and TimestampType are both inferred
    as type :py:data:`datetime <mlflow.types.DataType.datetime>`, which is coerced to
    TimestampType at inference.

    :param model_input: Valid input to the model. E.g. (a subset of) the training dataset.
    :param model_output: Valid model output. E.g. Model predictions for the (subset of) training
                         dataset.
    :return: ModelSignature
    """
    inputs = _infer_schema(model_input)
    outputs = _infer_schema(model_output) if model_output is not None else None
    return ModelSignature(inputs, outputs)


# `t\w*\.` matches the `typing` module or its alias
_LIST_OF_STRINGS_PATTERN = re.compile(r"^(t\w*\.)?list\[str\]$", re.IGNORECASE)


def _is_list_str(hint_str):
    return _LIST_OF_STRINGS_PATTERN.match(hint_str.replace(" ", "")) is not None


_LIST_OF_STR_DICT_PATTERN = re.compile(
    r"^(t\w*\.)?list\[(t\w*\.)?dict\[str,str\]\]$", re.IGNORECASE
)


def _is_list_of_string_dict(hint_str):
    return _LIST_OF_STR_DICT_PATTERN.match(hint_str.replace(" ", "")) is not None


def _infer_hint_from_str(hint_str):
    if _is_list_str(hint_str):
        return List[str]
    elif _is_list_of_string_dict(hint_str):
        return List[Dict[str, str]]
    else:
        return None


def _get_arg_names(f):
    return list(inspect.signature(f).parameters.keys())


class _TypeHints:
    def __init__(self, input_=None, output=None):
        self.input = input_
        self.output = output

    def __repr__(self):
        return "<input: {}, output: {}>".format(self.input, self.output)


def _extract_type_hints(f, input_arg_index):
    """
    Extract type hints from a function.

    :param f: Function to extract type hints from.
    :param input_arg_index: Index of the function argument that corresponds to the model input.
    :return: A `_TypeHints` object containing the input and output type hints.
    """
    if not hasattr(f, "__annotations__") and hasattr(f, "__call__"):
        return _extract_type_hints(f.__call__, input_arg_index)

    if f.__annotations__ == {}:
        return _TypeHints()

    arg_names = _get_arg_names(f)
    if len(arg_names) - 1 < input_arg_index:
        raise MlflowException.invalid_parameter_value(
            f"The specified input argument index ({input_arg_index}) is out of range for the "
            "function signature: {}".format(input_arg_index, arg_names)
        )
    arg_name = _get_arg_names(f)[input_arg_index]
    try:
        hints = get_type_hints(f)
    except TypeError:
        # ---
        # from __future__ import annotations # postpones evaluation of 'list[str]'
        #
        # def f(x: list[str]) -> list[str]:
        #          ^^^^^^^^^ Evaluating this expression ('list[str]') results in a TypeError in
        #                    Python < 3.9 because the built-in list type is not subscriptable.
        #     return x
        # ---
        # Best effort to infer type hints from strings
        hints = {}
        for arg in [arg_name, "return"]:
            if hint_str := f.__annotations__.get(arg, None):
                if hint := _infer_hint_from_str(hint_str):
                    hints[arg] = hint
                else:
                    _logger.info("Unsupported type hint: %s, skipping schema inference", hint_str)
    except Exception as e:
        _logger.warning("Failed to extract type hints from function %s: %s", f.__name__, repr(e))
        return _TypeHints()

    return _TypeHints(hints.get(arg_name), hints.get("return"))


def _infer_signature_from_type_hints(func, input_arg_index, input_example=None):
    hints = _extract_type_hints(func, input_arg_index)
    if hints.input is None:
        return None

    input_schema = _infer_schema_from_type_hint(hints.input, input_example) if hints.input else None
    output_example = func(input_example) if input_example else None
    output_schema = (
        _infer_schema_from_type_hint(hints.output, output_example) if hints.output else None
    )
    if input_schema is None and output_schema is None:
        return None
    return ModelSignature(inputs=input_schema, outputs=output_schema)
