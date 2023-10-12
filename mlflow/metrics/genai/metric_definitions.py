from typing import List, Optional

from mlflow.exceptions import MlflowException
from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.genai.genai_metric import make_genai_metric
from mlflow.metrics.genai.utils import _get_latest_metric_version
from mlflow.models import EvaluationMetric
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.class_utils import _get_class_from_string


def correctness(
    model: Optional[str] = None,
    metric_version: Optional[str] = None,
    examples: Optional[List[EvaluationExample]] = None,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the correctness of an LLM using the
    model provided. Correctness will be assessed by the similarity in meaning and description to
    the ground truth.

    The ground_truth variable must be provided as part of the input dataset or output predictions.
    This can be mapped to a column of a different name using the evaluator_config.

    An MlflowException will be raised if the specified version for this metric does not exist.

    :param model: (Optional) The model that will be used to evaluate this metric. Defaults to GPT-4.
    :param metric_version: (Optional) The version of the correctness metric to use.
        Defaults to the latest version.
    :param examples: (Optional) Provide a list of examples to help the judge model evaluate the
        correctness. It is highly recommended to add examples to be used as a reference to
        evaluate the new results.
    :return: A metric object
    """
    if metric_version is None:
        metric_version = _get_latest_metric_version()
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.CorrectnessMetric"
    try:
        correctness_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find correctness metric for version {metric_version}."
            f" Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct correctness metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = correctness_class_module.default_examples
    if model is None:
        model = correctness_class_module.default_model

    return make_genai_metric(
        name="correctness",
        definition=correctness_class_module.definition,
        grading_prompt=correctness_class_module.grading_prompt,
        examples=examples,
        version=metric_version,
        model=model,
        variables=correctness_class_module.variables,
        parameters=correctness_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )


def relevance(
    model: Optional[str] = None,
    metric_version: Optional[str] = None,
    examples: Optional[List[EvaluationExample]] = None,
) -> EvaluationMetric:
    """
    This function will create a genai metric used to evaluate the relevance of an LLM using the
    model provided. Relevance will be assessed by the appropriateness, significance, and
    applicability of the output with respect to the input and context.

    The context variable must be provided as part of the input dataset or output predictions.
    This can be mapped to a column of a different name using the evaluator_config.

    An MlflowException will be raised if the specified version for this metric does not exist.

    :param model: (Optional) The model that will be used to evaluate this metric. Defaults to GPT-4.
    :param metric_version: (Optional) The version of the relevance metric to use.
        Defaults to the latest version.
    :param examples: (Optional) Provide a list of examples to help the judge model evaluate the
        relevance. It is highly recommended to add examples to be used as a reference to evaluate
        the new results.
    :return: A metric object
    """
    if metric_version is None:
        metric_version = _get_latest_metric_version()
    class_name = f"mlflow.metrics.genai.prompts.{metric_version}.RelevanceMetric"
    try:
        relevance_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find relevance metric for version {metric_version}."
            f" Please check the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct relevance metric {metric_version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    if examples is None:
        examples = relevance_class_module.default_examples
    if model is None:
        model = relevance_class_module.default_model

    return make_genai_metric(
        name="relevance",
        definition=relevance_class_module.definition,
        grading_prompt=relevance_class_module.grading_prompt,
        examples=examples,
        version=metric_version,
        model=model,
        variables=relevance_class_module.variables,
        parameters=relevance_class_module.parameters,
        aggregations=["mean", "variance", "p90"],
        greater_is_better=True,
    )
