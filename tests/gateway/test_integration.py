import pytest
import os
import requests
from unittest.mock import patch

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.gateway import MlflowGatewayClient, query, set_gateway_uri, get_route
from mlflow.gateway.config import Route
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.providers.cohere import CohereProvider
from mlflow.gateway.providers.mlflow import MlflowModelServingProvider
import mlflow.gateway.utils
from mlflow.utils.request_utils import _cached_get_request_session

from tests.gateway.tools import UvicornGateway, save_yaml


@pytest.fixture
def basic_config_dict():
    return {
        "routes": [
            {
                "name": "chat-openai",
                "route_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "config": {"openai_api_key": "$OPENAI_API_KEY"},
                },
            },
            {
                "name": "completions-openai",
                "route_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4",
                    "provider": "openai",
                    "config": {"openai_api_key": "$OPENAI_API_KEY"},
                },
            },
            {
                "name": "embeddings-openai",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "openai",
                    "name": "text-embedding-ada-002",
                    "config": {
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_key": "$OPENAI_API_KEY",
                    },
                },
            },
            {
                "name": "completions-anthropic",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "anthropic",
                    "name": "claude-instant-1.1",
                    "config": {
                        "anthropic_api_key": "$ANTHROPIC_API_KEY",
                    },
                },
            },
            {
                "name": "completions-cohere",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "cohere",
                    "name": "command",
                    "config": {
                        "cohere_api_key": "$COHERE_API_KEY",
                    },
                },
            },
            {
                "name": "embeddings-cohere",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "cohere",
                    "name": "embed-english-v2.0",
                    "config": {
                        "cohere_api_key": "$COHERE_API_KEY",
                    },
                },
            },
            {
                "name": "chat-oss",
                "route_type": "llm/v1/chat",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "mpt-chatbot",
                    "config": {"mlflow_server_url": "http://127.0.0.1:5000"},
                },
            },
            {
                "name": "completions-oss",
                "route_type": "llm/v1/completions",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "mpt-completion-model",
                    "config": {"mlflow_server_url": "http://127.0.0.1:5001"},
                },
            },
            {
                "name": "embeddings-oss",
                "route_type": "llm/v1/embeddings",
                "model": {
                    "provider": "mlflow-model-serving",
                    "name": "sentence-transformers",
                    "config": {"mlflow_server_url": "http://127.0.0.1:5002"},
                },
            },
        ]
    }


@pytest.fixture(autouse=True)
def clear_uri():
    mlflow.gateway.utils._gateway_uri = None


@pytest.fixture
def gateway(basic_config_dict, tmp_path):
    conf = tmp_path / "config.yaml"
    save_yaml(conf, basic_config_dict)
    with UvicornGateway(conf) as g:
        yield g


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")


def test_create_gateway_client_with_declared_url(gateway):
    gateway_client = MlflowGatewayClient(gateway_uri=gateway.url)
    assert gateway_client.gateway_uri == gateway.url
    assert isinstance(gateway_client.get_route("chat-openai"), Route)
    routes = gateway_client.search_routes()
    assert len(routes) == 9
    assert all(isinstance(route, Route) for route in routes)


def test_openai_chat(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "test",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }

    data = {"messages": [{"role": "user", "content": "test"}]}

    async def mock_chat(self, payload):
        return expected_output

    with patch.object(OpenAIProvider, "chat", mock_chat):
        response = query(route=route.name, data=data)
    assert response == expected_output


def test_openai_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-openai")
    expected_output = {
        "candidates": [
            {
                "text": "test.",
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 4,
            "output_tokens": 7,
            "total_tokens": 11,
            "model": "gpt-4",
            "route_type": "llm/v1/completions",
        },
    }

    data = {"prompt": "test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(OpenAIProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_openai_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-openai")
    expected_output = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "metadata": {
            "input_tokens": 4,
            "output_tokens": 0,
            "total_tokens": 4,
            "model": "text-embedding-ada-002",
            "route_type": "llm/v1/embeddings",
        },
    }

    data = {"text": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(OpenAIProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_anthropic_completions(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("completions-anthropic")
    expected_output = {
        "candidates": [
            {
                "text": "test",
                "metadata": {"finish_reason": "length"},
            }
        ],
        "metadata": {
            "model": "claude-instant-1.1",
            "route_type": "llm/v1/completions",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    }

    data = {
        "prompt": "test",
        "max_tokens": 500,
        "temperature": 0.3,
    }

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(AnthropicProvider, "completions", mock_completions):
        response = query(route=route.name, data=data)
    assert response == expected_output


def test_cohere_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-cohere")
    expected_output = {
        "candidates": [
            {
                "text": "mock using MagicMock please",
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "model": "gpt-4",
            "route_type": "llm/v1/completions",
        },
    }

    data = {"prompt": "mock my test", "max_tokens": 50}

    async def mock_completions(self, payload):
        return expected_output

    with patch.object(CohereProvider, "completions", mock_completions):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_cohere_embeddings(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("embeddings-cohere")
    expected_output = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "metadata": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "model": "embed-english-v2.0",
            "route_type": "llm/v1/embeddings",
        },
    }

    data = {"text": "mock me and my test"}

    async def mock_embeddings(self, payload):
        return expected_output

    with patch.object(CohereProvider, "embeddings", mock_embeddings):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_invalid_response_structure_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
        "embeddings": [[0.0, 1.0]],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }

    data = {"messages": [{"role": "user", "content": "invalid test"}]}

    async def mock_chat(self, payload):
        return expected_output

    def _mock_request_session(max_retries, backoff_factor, retry_codes):
        return _cached_get_request_session(1, 1, retry_codes, os.getpid())

    with patch(
        "mlflow.utils.request_utils._get_request_session", _mock_request_session
    ), patch.object(OpenAIProvider, "chat", mock_chat), pytest.raises(
        MlflowException, match=".*Max retries exceeded.*"
    ):
        query(route=route.name, data=data)


def test_invalid_query_request_raises(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("chat-openai")
    expected_output = {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "test",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 17,
            "output_tokens": 24,
            "total_tokens": 41,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }

    data = {"text": "this is invalid"}

    async def mock_chat(self, payload):
        return expected_output

    def _mock_request_session(max_retries, backoff_factor, retry_codes):
        return _cached_get_request_session(2, 1, retry_codes, os.getpid())

    with patch(
        "mlflow.utils.request_utils._get_request_session", _mock_request_session
    ), patch.object(OpenAIProvider, "chat", new=mock_chat), pytest.raises(
        requests.exceptions.HTTPError, match="Unprocessable Entity for"
    ):
        query(route=route.name, data=data)


def test_mlflow_chat(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("chat-oss")
    expected_output = {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "test",
                },
                "metadata": {"finish_reason": "length"},
            }
        ],
        "metadata": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "model": "mpt-chatbot",
            "route_type": "llm/v1/chat",
        },
    }

    data = {"messages": [{"role": "user", "content": "test"}]}

    with patch.object(MlflowModelServingProvider, "chat", return_value=expected_output):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mlflow_completions(gateway):
    client = MlflowGatewayClient(gateway_uri=gateway.url)
    route = client.get_route("completions-oss")
    expected_output = {
        "candidates": [
            {
                "text": "test",
                "metadata": {"finish_reason": "length"},
            }
        ],
        "metadata": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "model": "mpt-completion-model",
            "route_type": "llm/v1/completions",
        },
    }

    data = {"prompt": "this is a test"}

    with patch.object(MlflowModelServingProvider, "completions", return_value=expected_output):
        response = client.query(route=route.name, data=data)
    assert response == expected_output


def test_mlflow_embeddings(gateway):
    set_gateway_uri(gateway_uri=gateway.url)
    route = get_route("embeddings-oss")
    expected_output = {
        "embeddings": [
            [0.001, -0.001],
            [0.002, -0.002],
        ],
        "metadata": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "model": "sentence-transformers",
            "route_type": "llm/v1/embeddings",
        },
    }

    data = {"text": ["test1", "test2"]}

    with patch.object(MlflowModelServingProvider, "embeddings", return_value=expected_output):
        response = query(route=route.name, data=data)
    assert response == expected_output
