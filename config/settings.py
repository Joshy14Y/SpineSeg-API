from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = {"env_file": ".env"}
    port: int = 8080
    weights_path: Path
    frontend_url: str


settings = Settings()
