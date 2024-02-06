import json
import logging

import numpy as np
import transformers

from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _contains_params
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.schema import ColSpec, DataType, Schema, TensorSpec
from mlflow.utils.process import _IS_UNIX
from mlflow.utils.timeout import MLflowTimeoutError, run_with_timeout

_logger = logging.getLogger(__name__)


_TEXT2TEXT_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string")]),
    outputs=Schema([ColSpec("string")]),
)
_CLASSIFICATION_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string")]),
    outputs=Schema([ColSpec("string", name="label"), ColSpec("double", name="score")]),
)


# Order is important here, the first matching pipeline type will be used
_DEFAULT_SIGNATURE_FOR_PIPELINES = {
    transformers.TokenClassificationPipeline: _TEXT2TEXT_SIGNATURE,
    transformers.ConversationalPipeline: _TEXT2TEXT_SIGNATURE,
    transformers.TranslationPipeline: _TEXT2TEXT_SIGNATURE,
    transformers.FillMaskPipeline: _TEXT2TEXT_SIGNATURE,
    transformers.TextGenerationPipeline: _TEXT2TEXT_SIGNATURE,
    transformers.Text2TextGenerationPipeline: _TEXT2TEXT_SIGNATURE,
    transformers.TextClassificationPipeline: _CLASSIFICATION_SIGNATURE,
    transformers.ImageClassificationPipeline: _CLASSIFICATION_SIGNATURE,
    transformers.ZeroShotClassificationPipeline: ModelSignature(
        inputs=Schema(
            [
                ColSpec(DataType.string, name="sequences"),
                ColSpec(DataType.string, name="candidate_labels"),
                ColSpec(DataType.string, name="hypothesis_template"),
            ]
        ),
        outputs=Schema(
            [
                ColSpec(DataType.string, name="sequence"),
                ColSpec(DataType.string, name="labels"),
                ColSpec(DataType.double, name="scores"),
            ]
        ),
    ),
    transformers.AutomaticSpeechRecognitionPipeline: ModelSignature(
        inputs=Schema([ColSpec(DataType.binary)]),
        outputs=Schema([ColSpec(DataType.string)]),
    ),
    transformers.AudioClassificationPipeline: ModelSignature(
        inputs=Schema([ColSpec(DataType.binary)]),
        outputs=Schema(
            [ColSpec(DataType.double, name="score"), ColSpec(DataType.string, name="label")]
        ),
    ),
    transformers.TableQuestionAnsweringPipeline: ModelSignature(
        inputs=Schema(
            [ColSpec(DataType.string, name="query"), ColSpec(DataType.string, name="table")]
        ),
        outputs=Schema([ColSpec(DataType.string)]),
    ),
    transformers.QuestionAnsweringPipeline: ModelSignature(
        inputs=Schema(
            [ColSpec(DataType.string, name="question"), ColSpec(DataType.string, name="context")]
        ),
        outputs=Schema([ColSpec(DataType.string)]),
    ),
    transformers.FeatureExtractionPipeline: ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]),
        outputs=Schema([TensorSpec(np.dtype("float64"), [-1], "double")]),
    ),
}


def infer_or_get_default_signature(
    pipeline, example=None, model_config=None, flavor_config=None, timeout=60
) -> ModelSignature:
    """
    Assigns a default ModelSignature for a given Pipeline type that has pyfunc support. These
    default signatures should only be generated and assigned when saving a model iff the user
    has not supplied a signature.
    For signature inference in some Pipelines that support complex input types, an input example
    is needed.
    """
    if example:
        try:
            return _infer_signature_with_prediction(
                pipeline, example, model_config, flavor_config, timeout
            )
        except MLflowTimeoutError:
            _logger.warning(
                "Attempted to generate a signature for the saved model or pipeline "
                "but prediction operation timed out. Falling back to the default "
                "signature for the pipeline type."
            )
            pass
        except Exception as e:
            _logger.error(
                "Attempted to generate a signature for the saved model or pipeline "
                f"but encountered an error: {e}"
            )
            raise

    for pipeline_type, signature in _DEFAULT_SIGNATURE_FOR_PIPELINES.items():
        if isinstance(pipeline, pipeline_type):
            return signature

    _logger.warning(
        "An unsupported Pipeline type was supplied for signature inference. "
        "Either provide an `input_example` or generate a signature manually "
        "via `infer_signature` if you would like to have a signature recorded "
        "in the MLmodel file."
    )


def _infer_signature_with_prediction(
    pipeline, example, model_config=None, flavor_config=None, timeout=300
) -> ModelSignature:
    import transformers

    params = None
    if _contains_params(example):
        example, params = example
    example = format_input_example_for_special_cases(example, pipeline)

    if not isinstance(pipeline, transformers.Pipeline):
        raise MlflowException(
            f"The pipeline type submitted is not a valid transformers Pipeline. "
            f"The type {type(pipeline).__name__} is not supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if _IS_UNIX:
        with run_with_timeout(timeout):
            prediction = _generate_signature_output(
                pipeline, example, model_config, flavor_config, params
            )
    else:
        _logger.warning(
            "Running prediction on the input example to infer model signature. On Windows, "
            "the prediction is not bound by a timeout and may hang indefinitely. If it "
            "hangs, please consider specifying the signature manually."
        )
        prediction = _generate_signature_output(
            pipeline, example, model_config, flavor_config, params
        )
    return infer_signature(example, prediction, params)


def format_input_example_for_special_cases(input_example, pipeline):
    """
    Handles special formatting for specific types of Pipelines so that the displayed example
    reflects the correct example input structure that mirrors the behavior of the input parsing
    for pyfunc.
    """
    import transformers

    input_data = input_example[0] if isinstance(input_example, tuple) else input_example

    if (
        isinstance(pipeline, transformers.ZeroShotClassificationPipeline)
        and isinstance(input_data, dict)
        and isinstance(input_data["candidate_labels"], list)
    ):
        input_data["candidate_labels"] = json.dumps(input_data["candidate_labels"])
    return input_data if not isinstance(input_example, tuple) else (input_data, input_example[1])


def _generate_signature_output(pipeline, data, model_config=None, flavor_config=None, params=None):
    import transformers

    # Lazy import to avoid circular dependencies. Ideally we should move _TransformersWrapper
    # out from __init__.py to avoid this.
    from mlflow.transformers import _TransformersWrapper

    if not isinstance(pipeline, transformers.Pipeline):
        raise MlflowException(
            f"The pipeline type submitted is not a valid transformers Pipeline. "
            f"The type {type(pipeline).__name__} is not supported.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return _TransformersWrapper(
        pipeline=pipeline, model_config=model_config, flavor_config=flavor_config
    ).predict(data, params=params)
