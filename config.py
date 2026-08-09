from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover
    from pydantic import BaseSettings

    class SettingsConfigDict(dict):
        """Fallback shim for environments without pydantic-settings."""

    HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    if HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
    else:

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

    app_name: str = "AI Utility API"
    app_version: str = "1.0.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    ai_base_url: str = "https://api.openai.com/v1"
    ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_timeout: float = 60.0
    ai_max_tokens: int = 1024
    ai_temperature: float = 0.7

    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.ai_service import AIService
    from app.utils.logger import logger

    app.state.ai_service = AIService()
    logger.info(
        "AI Utility API ready — model=%s provider=%s docs=%s",
        app.state.ai_service.model,
        app.state.ai_service.base_url,
        "/docs",
    )
    yield
    app.state.ai_service.close()
    logger.info("AI Utility API shut down")


def create_app() -> FastAPI:
    from app.routes.utility import router as utility_router

    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(utility_router, prefix=settings.api_prefix)

    @app.get("/", tags=["root"], include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    return app


app = create_app()
