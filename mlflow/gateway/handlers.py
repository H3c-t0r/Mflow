import pathlib
from enum import Enum
import json
import os
from pathlib import Path
from pydantic import BaseModel, validator, parse_obj_as
from pydantic.json import pydantic_encoder
from typing import Optional, Union, List, Dict, Any
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import PROVIDERS
from mlflow.gateway.utils import is_valid_endpoint_name, check_configuration_route_name_collisions
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class Provider(str, Enum):
    UNSPECIFIED_PROVIDER = "custom"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DATABRICKS_SERVING_ENDPOINT = "databricks_serving_endpoint"
    MLFLOW = "mlflow"


class RouteType(str, Enum):
    CUSTOM = "custom"
    LLM_V1_COMPLETIONS = "llm/v1/completions"
    LLM_V1_CHAT = "llm/v1/chat"


class OpenAIConfig(BaseModel):
    openai_api_key: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = "https://api.openai.com/v1"
    openai_api_version: Optional[str] = None
    openai_organization: Optional[str] = None


class AnthropicConfig(BaseModel):
    anthropic_api_key: Optional[str] = None
    anthropic_api_base: Optional[str] = "https://api.anthropic.com/"


class DatabricksConfig(BaseModel):
    databricks_api_token: Optional[str] = None
    databricks_api_base: str


class MLflowConfig(BaseModel):
    api_base: str


class CustomConfig(BaseModel):
    api_key: Optional[str] = None
    api_base: str
    api_version: Optional[str] = None


config_types = {
    Provider.OPENAI: OpenAIConfig,
    Provider.ANTHROPIC: AnthropicConfig,
    Provider.DATABRICKS_SERVING_ENDPOINT: DatabricksConfig,
    Provider.MLFLOW: MLflowConfig,
    Provider.UNSPECIFIED_PROVIDER: CustomConfig,
}


class ModelInfo(BaseModel):
    name: Optional[str] = None
    provider: Provider = Provider.UNSPECIFIED_PROVIDER


def _resolve_api_key_from_input(api_key_input):
    """
    Resolves the provided API key.

    Input formats accepted:

    - Path to a file as a string which will have the key loaded from it
    - environment variable name that stores the api key
    - the api key itself
    """

    if not isinstance(api_key_input, str):
        raise MlflowException(
            "The api key provided is not a string. Please provide either an environment "
            "variable key, a path to a file containing the api key, or the api key itself",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # try reading as an environment variable
    env_var_attempt = api_key_input[1:] if api_key_input.startswith("$") else api_key_input

    env_var = os.getenv(env_var_attempt)
    if env_var:
        return env_var

    # try reading from a local path
    file = pathlib.Path(api_key_input)
    if file.is_file():
        return file.read_text()

    # if the key itself is passed, return
    return api_key_input


def _extract_and_set_api_key(config, provider):
    required_keys = {
        OpenAIConfig: "openai_api_key",
        AnthropicConfig: "anthropic_api_key",
        DatabricksConfig: "databricks_api_token",
        CustomConfig: "api_key",
    }

    config_dict = config.dict()

    for config_class, key in required_keys.items():
        if isinstance(config, config_class):
            if getattr(config, key, None) is None:
                raise MlflowException(
                    f"For the {provider} provider, the api key must either be specified within the "
                    "configuration supplied or an environment variable set whose key is "
                    "defined within the configuration",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            else:
                # set the config key
                config_dict[key] = _resolve_api_key_from_input(config_dict[key])

    return config_types[provider](**config_dict)


def _validate_base_route(config, provider):
    base_route = {
        OpenAIConfig: "openai_api_base",
        AnthropicConfig: "anthropic_api_base",
        DatabricksConfig: "databricks_api_base",
        MLflowConfig: "mlflow_api_base",
        CustomConfig: "api_base",
    }

    for config_class, base in base_route.items():
        if isinstance(config, config_class) and getattr(config, base, None) is None:
            raise MlflowException(
                f"For the {provider} provider, the configuration is not set correctly. Verify "
                "that a config is set and that the base url and api key information is provided.",
                error_code=INVALID_PARAMETER_VALUE,
            )


# pylint: disable=no-self-argument
class Model(BaseModel):
    name: Optional[str] = None
    provider: Union[str, Provider] = Provider.UNSPECIFIED_PROVIDER
    config: Optional[Dict[str, Any]] = None

    @validator("provider", pre=True)
    def validate_provider(cls, value):
        if isinstance(value, Provider):
            return value
        return (
            Provider[value.upper()]
            if value.upper() in Provider.__members__
            else Provider.UNSPECIFIED_PROVIDER
        )

    @validator("config", pre=True)
    def validate_config(cls, config, values):
        provider = values.get("provider")
        if provider:
            config_type = config_types[provider]
            config_instance = config_type(**config)

            # set the api_key
            config_instance = _extract_and_set_api_key(config_instance, provider)

            # validate the base_route
            _validate_base_route(config_instance, provider)

            return config_instance
        else:
            raise MlflowException(
                "A provider must be provided for each gateway route.",
                error_code=INVALID_PARAMETER_VALUE,
            )


# pylint: disable=no-self-argument
class RouteConfig(BaseModel):
    name: str
    type: RouteType = RouteType.CUSTOM
    model: Model

    @validator("name")
    def validate_endpoint_name(cls, route_name):
        if not is_valid_endpoint_name(route_name):
            raise MlflowException(
                "The route name provided contains disallowed characters for a url endpoint. "
                f"'{route_name}' is invalid. Names cannot contain spaces or any non "
                "alphanumeric characters other than hyphen and underscore.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return route_name

    @validator("model", pre=True)
    def validate_model(cls, model):
        if model:
            model_instance = Model(**model)
            if model_instance.provider in PROVIDERS and model_instance.config is None:
                raise MlflowException(
                    "A config must be supplied when setting a provider. The provider entry for "
                    f"{model_instance.provider} is incorrect.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        return model

    @validator("type", pre=True)
    def validate_route_type(cls, value):
        if value in RouteType._value2member_map_:
            return value
        return RouteType.CUSTOM.value


class Route(BaseModel):
    name: str
    type: RouteType
    model: ModelInfo


def _load_route_config(path: Union[str, Path]) -> List[RouteConfig]:
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration RouteConfig class
    """
    if isinstance(path, str):
        path = Path(path)
    configuration = yaml.safe_load(path.read_text())
    check_configuration_route_name_collisions(configuration)
    return parse_obj_as(List[RouteConfig], configuration)


def _save_route_config(config: List[RouteConfig], path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    serialized = [
        json.loads(json.dumps(route.dict(), default=pydantic_encoder)) for route in config
    ]
    path.write_text(yaml.safe_dump(serialized))


def _route_config_to_route(route_config: RouteConfig) -> Route:
    return Route(
        name=route_config.name,
        type=route_config.type,
        model=ModelInfo(
            name=route_config.model.name,
            provider=route_config.model.provider,
        ),
    )


def _route_configs_to_routes(route_config: List[RouteConfig]) -> List[Route]:
    return [_route_config_to_route(route) for route in route_config]
