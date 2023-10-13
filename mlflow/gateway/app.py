import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from mlflow.environment_variables import MLFLOW_GATEWAY_CONFIG
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import SetLimitsModel
from mlflow.gateway.config import (
    GatewayConfig,
    LimitsConfig,
    Route,
    RouteConfig,
    RouteType,
    _load_route_config,
)
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_CRUD_ROUTE_BASE,
    MLFLOW_GATEWAY_HEALTH_ENDPOINT,
    MLFLOW_GATEWAY_LIMITS_BASE,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE,
    MLFLOW_QUERY_SUFFIX,
)
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import SearchRoutesToken
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)


class GatewayAPI(FastAPI):
    def __init__(self, config: GatewayConfig, limiter: Limiter, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dynamic_routes: Dict[str, Route] = {}
        self.state.limiter = limiter
        self.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.set_dynamic_routes(config, limiter)

    def set_dynamic_routes(self, config: GatewayConfig, limiter: Limiter) -> None:
        self.dynamic_routes.clear()
        for route in config.routes:
            self.add_api_route(
                path=f"{MLFLOW_GATEWAY_ROUTE_BASE}{route.name}{MLFLOW_QUERY_SUFFIX}",
                endpoint=_route_type_to_endpoint(route, limiter),
                methods=["POST"],
            )
            self.dynamic_routes[route.name] = route.to_route()

    def get_dynamic_route(self, route_name: str) -> Optional[Route]:
        return self.dynamic_routes.get(route_name)


def _create_chat_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    # https://slowapi.readthedocs.io/en/latest/#limitations-and-known-issues
    async def _chat(request: Request) -> chat.ResponsePayload:
        payload = await request.json()
        return await prov.chat(chat.RequestPayload(**payload))

    return _chat


def _create_completions_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _completions(request: Request) -> completions.ResponsePayload:
        payload = await request.json()
        return await prov.completions(**payload)

    return _completions


def _create_embeddings_endpoint(config: RouteConfig):
    prov = get_provider(config.model.provider)(config)

    async def _embeddings(request: Request) -> embeddings.ResponsePayload:
        payload = await request.json()
        return await prov.embeddings(embeddings.RequestPayload(**payload))

    return _embeddings


async def _custom(request: Request):
    return request.json()


def _route_type_to_endpoint(config: RouteConfig, limiter: Limiter):
    provider_to_factory = {
        RouteType.LLM_V1_CHAT: _create_chat_endpoint,
        RouteType.LLM_V1_COMPLETIONS: _create_completions_endpoint,
        RouteType.LLM_V1_EMBEDDINGS: _create_embeddings_endpoint,
    }
    if factory := provider_to_factory.get(config.route_type):
        handler = factory(config)
        if config.limit:
            limit_value = f"{config.limit.calls}/{config.limit.renewal_period}"
            return limiter.limit(limit_value)(handler)
        else:
            return handler

    raise HTTPException(
        status_code=404,
        detail=f"Unexpected route type {config.route_type!r} for route {config.name!r}.",
    )


class HealthResponse(BaseModel):
    status: str


class SearchRoutesResponse(BaseModel):
    routes: List[Route]
    next_page_token: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "routes": [
                    {
                        "name": "openai-chat",
                        "route_type": "llm/v1/chat",
                        "model": {
                            "name": "gpt-3.5-turbo",
                            "provider": "openai",
                        },
                    },
                    {
                        "name": "anthropic-completions",
                        "route_type": "llm/v1/completions",
                        "model": {
                            "name": "claude-instant-100k",
                            "provider": "anthropic",
                        },
                    },
                    {
                        "name": "cohere-embeddings",
                        "route_type": "llm/v1/embeddings",
                        "model": {
                            "name": "embed-english-v2.0",
                            "provider": "cohere",
                        },
                    },
                ],
                "next_page_token": "eyJpbmRleCI6IDExfQ==",
            }
        }


def create_app_from_config(config: GatewayConfig) -> GatewayAPI:
    """
    Create the GatewayAPI app from the gateway configuration.
    """
    limiter = Limiter(key_func=get_remote_address)
    app = GatewayAPI(
        config=config,
        limiter=limiter,
        title="MLflow Gateway API",
        description="The core gateway API for reverse proxy interface using remote inference "
        "endpoints within MLflow",
        version=VERSION,
        docs_url=None,
    )

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        for directory in ["build", "public"]:
            favicon = Path(__file__).parent.parent.joinpath(
                "server", "js", directory, "favicon.ico"
            )
            if favicon.exists():
                return FileResponse(favicon)
        raise HTTPException(status_code=404, detail="favicon.ico not found")

    @app.get("/docs", include_in_schema=False)
    async def docs():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="MLflow Gateway API",
            swagger_favicon_url="/favicon.ico",
        )

    @app.get(MLFLOW_GATEWAY_HEALTH_ENDPOINT)
    async def health() -> HealthResponse:
        return {"status": "OK"}

    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE + "{route_name}")
    async def get_route(route_name: str) -> Route:
        if matched := app.get_dynamic_route(route_name):
            return matched

        raise HTTPException(
            status_code=404,
            detail=f"The route '{route_name}' is not present or active on the server. Please "
            "verify the route name.",
        )

    @app.get(MLFLOW_GATEWAY_CRUD_ROUTE_BASE)
    async def search_routes(page_token: Optional[str] = None) -> SearchRoutesResponse:
        start_idx = SearchRoutesToken.decode(page_token).index if page_token is not None else 0

        end_idx = start_idx + MLFLOW_GATEWAY_SEARCH_ROUTES_PAGE_SIZE
        routes = list(app.dynamic_routes.values())
        result = {"routes": routes[start_idx:end_idx]}
        if len(routes[end_idx:]) > 0:
            next_page_token = SearchRoutesToken(index=end_idx)
            result["next_page_token"] = next_page_token.encode()

        return result

    @app.get(MLFLOW_GATEWAY_LIMITS_BASE + "{route}")
    async def get_limits(route: str) -> LimitsConfig:
        raise HTTPException(
            status_code=501, detail="The get_limits API is not available in OSS MLflow AI Gateway."
        )

    @app.post(MLFLOW_GATEWAY_LIMITS_BASE)
    async def set_limits(payload: SetLimitsModel) -> LimitsConfig:
        raise HTTPException(
            status_code=501, detail="The set_limits API is not available in OSS MLflow AI Gateway."
        )

    return app


def create_app_from_path(config_path: Union[str, Path]) -> GatewayAPI:
    """
    Load the path and generate the GatewayAPI app instance.
    """
    config = _load_route_config(config_path)
    return create_app_from_config(config)


def create_app_from_env() -> GatewayAPI:
    """
    Load the path from the environment variable and generate the GatewayAPI app instance.
    """
    if config_path := MLFLOW_GATEWAY_CONFIG.get():
        return create_app_from_path(config_path)

    raise MlflowException(
        f"Environment variable {MLFLOW_GATEWAY_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )
