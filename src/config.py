"""Configuration management for the Policy RAG Agent."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Policy RAG Agent"
    app_version: str = "0.2.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout_seconds: int = 30

    # LLM
    llm_provider: Literal["mock", "openai", "anthropic", "azure"] = "mock"
    llm_model: str = "gpt-4"
    llm_api_key: str = ""
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.1

    # RAG
    rag_top_k: int = 4
    rag_min_confidence: float = 0.25
    rag_citation_format: str = "[{id}]"

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # Security
    cors_origins: str = "*"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Monitoring
    enable_metrics: bool = True
    enable_tracing: bool = False
    sentry_dsn: str = ""

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
