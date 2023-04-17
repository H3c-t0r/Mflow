import json
import requests
from unittest import mock
from contextlib import contextmanager


TEST_CONTENT = "test"


class _MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.content = json.dumps(json_data).encode()
        self.headers = {"Content-Type": "application/json"}


def _chat_completion_json_sample():
    # https://platform.openai.com/docs/api-reference/chat/create
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": TEST_CONTENT},
                "finish_reason": "stop",
                "text": TEST_CONTENT,
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def _models_retrieve_json_sample():
    # https://platform.openai.com/docs/api-reference/models/retrieve
    return {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "owned_by": "openai",
        "permission": [],
    }


def _mock_chat_completion_response():
    return _MockResponse(200, _chat_completion_json_sample())


def _mock_models_retrieve_response():
    return _MockResponse(200, _models_retrieve_json_sample())


@contextmanager
def _mock_request(**kwargs):
    with mock.patch("requests.Session.request", **kwargs) as m:
        yield m


def _mock_chat_completion_request():
    original = requests.Session.request

    def request(*args, **kwargs):
        if len(args) > 2:
            url = args[2]
        else:
            url = kwargs.get("url")

        if "chat/completions" in url:
            return _mock_chat_completion_response()
        else:
            return original(*args, **kwargs)

    return _mock_request(new=request)
