"""
Centralized application configuration.

All configuration is loaded from environment variables (via a .env file in
local development, or real environment variables in Docker/production).
Nothing here is hardcoded -- no passwords, no secret keys, no URLs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- App ---
    app_name: str = Field(default="FairCredit AI", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    # --- Database ---
    mysql_host: str = Field(default="mysql", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_database: str = Field(default="faircredit_ai", alias="MYSQL_DATABASE")
    mysql_user: str = Field(default="faircredit_user", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    database_url: str = Field(default="", alias="DATABASE_URL")

    # --- Backend ---
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        alias="CORS_ORIGINS",
    )

    # --- Frontend ---
    backend_url: str = Field(default="http://backend:8000", alias="BACKEND_URL")

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlalchemy_database_url(self) -> str:
        """
        Build the SQLAlchemy connection string.
        Prefers an explicit DATABASE_URL if provided, otherwise assembles
        one from the individual MySQL settings.
        """
        if self.database_url:
            return self.database_url
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance -- reads the environment once per process."""
    return Settings()
