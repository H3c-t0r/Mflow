from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from mlflow.transformers.hub_utils import get_latest_commit_for_repo
from mlflow.transformers.peft import _PEFT_ADAPTOR_DIR_NAME, get_peft_base_model, is_peft_model
from mlflow.transformers.torch_utils import _extract_torch_dtype_if_set

if TYPE_CHECKING:
    import transformers


# Flavor configuration keys
class FlavorKey:
    TASK = "task"
    INSTANCE_TYPE = "instance_type"
    TORCH_DTYPE = "torch_dtype"
    FRAMEWORK = "framework"

    MODEL = "model"
    MODEL_TYPE = "pipeline_model_type"
    MODEL_BINARY = "model_binary"
    MODEL_NAME = "source_model_name"
    MODEL_REVISION = "source_model_revision"

    PEFT = "peft_adaptor"

    COMPONENTS = "components"
    COMPONENT_NAME = "{}_name"  # e.g. tokenizer_name
    COMPONENT_REVISION = "{}_revision"
    COMPONENT_TYPE = "{}_type"
    TOKENIZER = "tokenizer"
    FEATURE_EXTRACTOR = "feature_extractor"
    IMAGE_PROCESSOR = "image_processor"
    PROCESSOR = "processor"
    PROCESSOR_TYPE = "processor_type"

    PROMPT_TEMPLATE = "prompt_template"


def build_flavor_config(
    pipeline: transformers.Pipeline, processor=None, save_pretrained=True
) -> Dict[str, Any]:
    """
    Generates the base flavor metadata needed for reconstructing a pipeline from saved
    components. This is important because the ``Pipeline`` class does not have a loader
    functionality. The serialization of a Pipeline saves the model, configurations, and
    metadata for ``FeatureExtractor``s, ``Processor``s, and ``Tokenizer``s exclusively.
    This function extracts key information from the submitted model object so that the precise
    instance types can be loaded correctly.

    Args:
        pipeline: Transformer pipeline to generate the flavor configuration for.
        processor: Optional processor instance to save alongside the pipeline.
        save_pretrained: Whether to save the pipeline and components weights to local disk.

    Returns:
        A dictionary containing the flavor configuration for the pipeline and its components,
        i.e. the configurations stored in "transformers" key in the MLModel YAML file.
    """
    flavor_conf = _generate_base_config(pipeline)

    if is_peft_model(pipeline.model):
        flavor_conf[FlavorKey.PEFT] = _PEFT_ADAPTOR_DIR_NAME
        model = get_peft_base_model(pipeline.model)
    else:
        model = pipeline.model

    flavor_conf.update(_get_model_config(model, save_pretrained))

    components = _get_components_from_pipeline(pipeline, processor)
    for key, instance in components.items():
        # Some components don't have name_or_path, then we fallback to the one from the model.
        flavor_conf.update(
            _get_component_config(instance, key, save_pretrained, default_repo=model.name_or_path)
        )

    # "components" field doesn't include processor
    components.pop(FlavorKey.PROCESSOR, None)
    flavor_conf[FlavorKey.COMPONENTS] = list(components.keys())

    return flavor_conf


def _generate_base_config(pipeline):
    flavor_conf = {
        FlavorKey.TASK: pipeline.task,
        FlavorKey.INSTANCE_TYPE: _get_instance_type(pipeline),
    }

    if framework := getattr(pipeline, "framework", None):
        flavor_conf[FlavorKey.FRAMEWORK] = framework

    # Extract a serialized representation of torch_dtype if provided
    if torch_dtype := _extract_torch_dtype_if_set(pipeline):
        # Convert the torch dtype and back to standardize the string representation
        flavor_conf[FlavorKey.TORCH_DTYPE] = str(torch_dtype)

    return flavor_conf


def _get_model_config(model, save_pretrained=True):
    conf = {
        FlavorKey.MODEL_TYPE: _get_instance_type(model),
        FlavorKey.MODEL_NAME: model.name_or_path,
    }

    if save_pretrained:
        # log local path to model binary file
        from mlflow.transformers.model_io import _MODEL_BINARY_FILE_NAME

        conf[FlavorKey.MODEL_BINARY] = _MODEL_BINARY_FILE_NAME
    else:
        # log HuggingFace repo name and commit hash
        conf[FlavorKey.MODEL_REVISION] = get_latest_commit_for_repo(model.name_or_path)

    return conf


def _get_component_config(component, key, save_pretrained=True, default_repo=None):
    conf = {FlavorKey.COMPONENT_TYPE.format(key): _get_instance_type(component)}

    # Log source repo name and commit sha for the component
    if not save_pretrained:
        repo = getattr(component, "name_or_path", default_repo)
        revision = get_latest_commit_for_repo(repo)
        conf[FlavorKey.COMPONENT_NAME.format(key)] = repo
        conf[FlavorKey.COMPONENT_REVISION.format(key)] = revision

    return conf


def _get_components_from_pipeline(pipeline, processor=None):
    supported_component_names = [
        FlavorKey.FEATURE_EXTRACTOR,
        FlavorKey.TOKENIZER,
        FlavorKey.IMAGE_PROCESSOR,
    ]

    components = {}
    for name in supported_component_names:
        if instance := getattr(pipeline, name, None):
            components[name] = instance

    if processor:
        components[FlavorKey.PROCESSOR] = processor

    return components


def _get_instance_type(obj):
    """
    Utility for extracting the saved object type or, if the `base` argument is set to `True`,
    the base ABC type of the model.
    """
    return obj.__class__.__name__
