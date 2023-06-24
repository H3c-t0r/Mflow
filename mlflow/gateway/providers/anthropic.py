from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from .base import BaseProvider
from .utils import send_request, rename_payload_keys
from ..config import AnthropicConfig, RouteConfig
from ..schemas import completions, chat, embeddings


class AnthropicProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, AnthropicConfig):
            raise TypeError(f"Invalid config type {config.model.config}")
        self.anthropic_config: AnthropicConfig = config.model.config
        self.headers = {"x-api-key": self.anthropic_config.anthropic_api_key}
        self.base_url = self.anthropic_config.anthropic_api_base

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        if "top_p" in payload:
            raise HTTPException(
                status_code=400,
                detail="Cannot set both 'temperature' and 'top_p' parameters. "
                "Please use only the temperature parameter for your query.",
            )
        if payload["max_tokens"] is None:
            raise HTTPException(
                status_code=400,
                detail="You must set an integer value for 'max_tokens' for the Anthropic provider "
                "that provides the upper bound on the returned token count.",
            )
        if payload.get("stream", None) == "true":
            raise HTTPException(
                status_code=400,
                detail="Setting the 'stream' parameter to 'true' is not supported with the MLflow "
                "Gateway.",
            )

        payload = rename_payload_keys(
            payload, {"max_tokens": "max_tokens_to_sample", "stop": "stop_sequences"}
        )

        payload["prompt"] = f"\n\nHuman: {payload['prompt']}\n\nAssistant:"

        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="complete",
            payload={"model": self.config.model.name, **payload},
        )

        # Example response:
        # Documentation: https://docs.anthropic.com/claude/reference/complete_post
        # ```
        # {
        #     "completion": " Hello! My name is Claude."
        #     "stop_reason": "stop_sequence",
        #     "model": "claude-instant-1.1",
        #     "truncated": False,
        #     "stop": None,
        #     "log_id": "dee173f87ddf1357da639dee3c38d833",
        #     "exception": None,
        # }
        # ```

        stop_reason = "stop" if resp["stop_reason"] == "stop_sequence" else "length"

        return completions.ResponsePayload(
            **{
                "candidates": [
                    {"text": resp["completion"], "metadata": {"finish_reason": stop_reason}}
                ],
                "metadata": {
                    "model": resp["model"],
                    "route_type": self.config.route_type,
                },
            }
        )

    async def chat(self, payload: chat.RequestPayload) -> None:
        # Anthropic does not have a chat endpoint
        raise HTTPException(
            status_code=404, detail="The chat route is not available for Anthropic models."
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> None:
        # Anthropic does not have an embeddings endpoint
        raise HTTPException(
            status_code=404, detail="The embeddings route is not available for Anthropic models."
        )
