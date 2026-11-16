"""
Application configuration management.
Handles environment variables, local Ollama settings, and database configuration.
"""

from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find .env file (project root)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: load from current directory
    load_dotenv(".env")


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Stock Backtesting Dashboard"

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Database settings (future use)
    DATABASE_URL: Optional[str] = None

    # External API settings
    YFINANCE_TIMEOUT: int = 30

    # Backtesting settings
    DEFAULT_INITIAL_CAPITAL: float = 10000.0
    DEFAULT_COMMISSION: float = 0.001

    # Local Qwen inference through the existing Ollama installation.
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen3:4b"
    OLLAMA_NUM_CTX: int = Field(default=8192, ge=1)
    OLLAMA_NUM_THREAD: int = Field(default=4, ge=1)
    OLLAMA_KEEP_ALIVE: str = "30m"
    OLLAMA_THINK: bool = False
    OLLAMA_TIMEOUT_SECONDS: float = Field(default=0, ge=0)

    # Additional .env settings
    REDIS_URL: Optional[str] = None
    BACKEND_URL: Optional[str] = None
    FRONTEND_URL: Optional[str] = None
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
