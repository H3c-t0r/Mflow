from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
import logging
import os
from typing import Any, Optional, Dict

from mlflow.version import VERSION
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_HEALTH_ENDPOINT,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_QUERY_SUFFIX,
)
from mlflow.gateway.config import (
    Route,
    RouteConfig,
    RouteType,
    GatewayConfig,
    _load_route_config,
)
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.providers import get_provider

_logger = logging.getLogger(__name__)

MLFLOW_GATEWAY_CONFIG = "MLFLOW_GATEWAY_CONFIG"


class GatewayAPI(FastAPI):
    def __init__(self, config: GatewayConfig, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dynamic_routes: Dict[str, Route] = {}
        self.set_dynamic_routes(config)

    def set_dynamic_routes(self, config: GatewayConfig) -> None:
        self.dynamic_routes.clear()
        for route in config.routes:
            self.add_api_route(
                path=f"{MLFLOW_GATEWAY_ROUTE_BASE}{route.name}{MLFLOW_QUERY_SUFFIX}",
                endpoint=_route_type_to_endpoint(route),
                methods=["POST"],
            )
            self.dynamic_routes[route.name] = route.to_route()

    def get_dynamic_route(self, route_name: str) -> Optional[Route]:
        return self.dynamic_routes.get(route_name)


def _create_chat_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _chat(payload: chat.RequestPayload) -> chat.ResponsePayload:
        return await prov.chat(payload)

    return _chat


def _create_completions_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _completions(
        payload: completions.RequestPayload,
    ) -> completions.ResponsePayload:
        return await prov.completions(payload)

    return _completions


def _create_embeddings_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _embeddings(payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        return await prov.embeddings(payload)

    return _embeddings


async def _custom(request: Request):
    return request.json()


def _route_type_to_endpoint(config: RouteConfig):
    provider_to_factory = {
        RouteType.LLM_V1_CHAT: _create_chat_endpoint,
        RouteType.LLM_V1_COMPLETIONS: _create_completions_endpoint,
        RouteType.LLM_V1_EMBEDDINGS: _create_embeddings_endpoint,
    }
    if factory := provider_to_factory.get(config.route_type):
        return factory(config)

    raise HTTPException(
        status_code=404,
        detail=f"Unexpected route type {config.route_type!r} for route {config.name!r}.",
    )


def create_app_from_config(config: GatewayConfig) -> GatewayAPI:
    """
    Create the GatewayAPI app from the gateway configuration.
    """
    app = GatewayAPI(
        config=config,
        title="MLflow Gateway API",
        description="The core gateway API for reverse proxy interface using remote inference "
        "endpoints within MLflow",
        version=VERSION,
    )

    @app.get("/")
    async def index():
        return RedirectResponse(url="/docs")

    @app.get(MLFLOW_GATEWAY_HEALTH_ENDPOINT)
    async def health():
        return {"status": "OK"}

    @app.get(MLFLOW_GATEWAY_ROUTE_BASE + "{route_name}")
    async def get_route(route_name: str):
        if matched := app.get_dynamic_route(route_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. Please "
            "verify the route name.",
        )

    @app.get(MLFLOW_GATEWAY_ROUTE_BASE)
    async def search_routes():
        # placeholder route listing functionality

        return {"routes": list(app.dynamic_routes.values())}

    return app


def create_app_from_env() -> GatewayAPI:
    """
    Load the path from the environment variable and generate the GatewayAPI app instance.
    """
    if config_path := os.getenv(MLFLOW_GATEWAY_CONFIG):
        config = _load_route_config(config_path)
        return create_app_from_config(config)

    raise MlflowException(
        f"Environment variable {MLFLOW_GATEWAY_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )
