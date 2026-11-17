"""
Application configuration management.
Handles environment variables, local Ollama settings, and database configuration.
"""

from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

# Load environment variables
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find .env file (project root)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: load from current directory
    load_dotenv(".env")


def local_ollama_url(value: str) -> str:
    """Accept only a loopback Ollama origin on the same machine."""
    url = value.strip().rstrip("/")
    parts = urlsplit(url)
    host = parts.hostname
    if host == "localhost":
        loopback = True
    else:
        try:
            loopback = ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if (
        not loopback
        or parts.scheme not in ("http", "https")
        or parts.username is not None
        or parts.password is not None
        or parts.path
        or parts.query
        or parts.fragment
    ):
        raise ValueError("OLLAMA_BASE_URL must be a local loopback origin.")
    return url


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Stock Backtesting Dashboard"

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # External API settings
    YFINANCE_TIMEOUT: int = 30

    # Backtesting settings
    DEFAULT_INITIAL_CAPITAL: float = 10000.0
    DEFAULT_COMMISSION: float = 0.001
    PIPELINE_DB_PATH: str = str(env_path.parent / "data" / "stage-gates.sqlite3")
    PAPER_DB_PATH: str = str(env_path.parent / "data" / "paper-trader.sqlite3")
    PIPELINE_LOCK_PATH: str = str(env_path.parent / "data" / "stage-work.lock")
    PIPELINE_NUM_THREADS: int = Field(default=4, ge=1)

    MEMORY_CHUNK_CHARS: int = Field(default=768, ge=128, le=4096)
    MEMORY_RECENT_EPISODES: int = Field(default=6, ge=0, le=50)

    # Local Qwen inference through the existing Ollama installation.
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen3:4b"
    OLLAMA_NUM_CTX: int = Field(default=8192, ge=1)
    OLLAMA_NUM_THREAD: int = Field(default=4, ge=1)
    OLLAMA_KEEP_ALIVE: str = "30m"
    OLLAMA_THINK: bool = False
    OLLAMA_TIMEOUT_SECONDS: float = Field(default=0, ge=0)

    @field_validator("OLLAMA_BASE_URL")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        return local_ollama_url(value)

    # Separate additive intelligence archive; portfolio databases are untouched.
    INTELLIGENCE_ENABLED: bool = True
    INTELLIGENCE_DB_PATH: str = str(env_path.parent / "data" / "market-intelligence.sqlite3")
    INTELLIGENCE_NETWORK_JOBS: int = Field(default=2, ge=1, le=16)
    INTELLIGENCE_MAX_BYTES: int = Field(default=2000000, ge=1024, le=10000000)
    INTELLIGENCE_USER_AGENT: str = "TomeyGeneralCompany-MarketResearch/1.0"
    INTELLIGENCE_FEEDS: str = "[]"
    INTELLIGENCE_BENCHMARKS: str = "SPY,^GSPC,XLK,XLF,XLE,XLI,XLU,CL=F,GC=F,^TNX,EURUSD=X"
    INTELLIGENCE_FRED_SERIES: str = "DFF"
    SEC_USER_AGENT: str = ""
    FRED_API_KEY: str = ""

    # Additional .env settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
