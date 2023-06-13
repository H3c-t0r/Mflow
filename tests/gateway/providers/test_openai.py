from unittest import mock

from fastapi.encoders import jsonable_encoder
import pytest

from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.config import RouteConfig


@pytest.mark.asyncio
async def test_chat():
    resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }
    config = RouteConfig(
        **{
            "name": "chat",
            "type": "llm/v1/chat",
            "model": {
                "provider": "openai",
                "name": "gpt-3.5-turbo",
                "config": {
                    "openai_api_base": "https://api.openai.com/v1",
                    "openai_api_key": "$OPENAI_API_KEY",
                },
            },
        }
    )
    with mock.patch("openai.ChatCompletion.acreate", return_value=resp) as mock_acreate:
        provider = OpenAIProvider(config)
        payload = {"messages": [{"role": "user", "content": "Tell me a joke"}]}
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "\n\nThis is a test!",
                    },
                    "metadata": {
                        "finish_reason": "stop",
                    },
                }
            ],
            "metadata": {
                "input_tokens": 13,
                "output_tokens": 7,
                "total_tokens": 20,
                "model": "gpt-3.5-turbo-0301",
                "route_type": "llm/v1/chat",
            },
        }
        mock_acreate.assert_called_once()


@pytest.mark.asyncio
async def test_completions():
    resp = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    config = {
        "name": "completions",
        "type": "llm/v1/completions",
        "model": {
            "provider": "openai",
            "name": "text-davinci-003",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "$OPENAI_API_KEY",
            },
        },
    }
    with mock.patch("openai.Completion.acreate", return_value=resp) as mock_acreate:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {"text": "\n\nThis is indeed a test", "metadata": {"finish_reason": "length"}}
            ],
            "metadata": {
                "input_tokens": 5,
                "output_tokens": 7,
                "total_tokens": 12,
                "model": "text-davinci-003",
                "route_type": "llm/v1/completions",
            },
        }
        mock_acreate.assert_called_once()


@pytest.mark.asyncio
async def test_embeddings():
    resp = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
    config = {
        "name": "embeddings",
        "type": "llm/v1/embeddings",
        "model": {
            "provider": "openai",
            "name": "text-embedding-ada-002",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "$OPENAI_API_KEY",
            },
        },
    }
    with mock.patch("openai.Embedding.acreate", return_value=resp) as mock_acreate:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {"text": "This is a test"}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "embeddings": [0.0023064255, -0.009327292, -0.0028842222],
            "metadata": {
                "input_tokens": 8,
                "output_tokens": 0,
                "total_tokens": 8,
                "model": "text-embedding-ada-002",
                "route_type": "llm/v1/embeddings",
            },
        }
        mock_acreate.assert_called_once()
