"""Shared local Ollama client for trading decisions and analysis."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Literal, Optional

import httpx

from app.config import settings
from app.llm.memory import prompt_budget
from app.stage_resources import StageResourceError, inference_resources

# All client instances share the installation's heavy-work resource slot.
# This includes backtests, retrospective analysis, and daily feedback.


class OllamaError(RuntimeError):
    """A local model request failed or did not return a complete answer."""


@dataclass(frozen=True)
class LLMResponse:
    """Keep the response.content interface used by the application."""

    content: str


@dataclass
class LLMClientConfig:
    """Local model configuration, with environment-backed defaults."""

    provider: Literal["ollama"] = "ollama"
    model: str = field(default_factory=lambda: settings.OLLAMA_MODEL)
    base_url: str = field(default_factory=lambda: settings.OLLAMA_BASE_URL)
    temperature: float = 0.1
    max_tokens: int = 4000
    num_ctx: int = field(default_factory=lambda: settings.OLLAMA_NUM_CTX)
    num_thread: int = field(default_factory=lambda: settings.OLLAMA_NUM_THREAD)
    keep_alive: str = field(default_factory=lambda: settings.OLLAMA_KEEP_ALIVE)
    think: bool = field(default_factory=lambda: settings.OLLAMA_THINK)
    timeout_seconds: float = field(
        default_factory=lambda: settings.OLLAMA_TIMEOUT_SECONDS
    )
    response_format: Optional[Literal["json"]] = None

    def __post_init__(self) -> None:
        if self.provider != "ollama":
            raise ValueError("Only the local Ollama provider is supported.")
        self.model = self.model.strip()
        self.base_url = self.base_url.strip().rstrip("/")
        if not self.model or not self.base_url:
            raise ValueError("Ollama model and base URL must not be empty.")
        if min(self.max_tokens, self.num_ctx, self.num_thread) < 1:
            raise ValueError("Token, context, and thread limits must be positive.")
        if self.timeout_seconds < 0:
            raise ValueError("Ollama timeout must be zero or positive.")
        if self.response_format not in (None, "json"):
            raise ValueError("Ollama response format must be 'json' or None.")

    def create_client(self) -> "OllamaClient":
        """Create a client without making a network request or loading a model."""
        return OllamaClient(self)


class OllamaClient:
    """Call Ollama's native chat endpoint; no cloud credentials are used."""

    def __init__(self, config: LLMClientConfig):
        self.config = config

    def invoke(self, prompt: str) -> LLMResponse:
        """Return the completed answer, excluding Ollama's separate thinking field."""
        config = self.config
        if len(prompt.encode("utf-8")) > prompt_budget(
            config.num_ctx, config.max_tokens
        ):
            raise OllamaError(
                "Prompt exceeds the context budget. Retrieve fewer evidence chunks or increase OLLAMA_NUM_CTX."
            )
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": config.think,
            "keep_alive": config.keep_alive,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "num_ctx": config.num_ctx,
                "num_thread": config.num_thread,
            },
        }
        if config.response_format:
            payload["format"] = config.response_format

        # Zero means no read deadline for slow CPU inference. Connection and
        # write failures still have finite timeouts and actionable errors.
        timeout = httpx.Timeout(
            config.timeout_seconds or None, connect=5.0, write=30.0, pool=5.0
        )
        try:
            with inference_resources():
                # Ignore proxy environment variables for this local connection.
                with httpx.Client(timeout=timeout, trust_env=False) as client:
                    response = client.post(f"{config.base_url}/api/chat", json=payload)
                    response.raise_for_status()
        except StageResourceError as exc:
            raise OllamaError(str(exc)) from exc
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Cannot connect to Ollama at {config.base_url}. "
                "Start Ollama with 'ollama serve' and check OLLAMA_BASE_URL."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(
                "Ollama request timed out. Check that Ollama is running; "
                "increase OLLAMA_TIMEOUT_SECONDS or set it to 0 for no read deadline."
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaError(
                    f"Ollama model '{config.model}' was not found. "
                    f"Run 'ollama pull {config.model}' and check OLLAMA_BASE_URL."
                ) from exc
            raise OllamaError(
                f"Ollama returned HTTP {exc.response.status_code}. "
                "Check the local Ollama server logs."
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaError(f"Local Ollama request failed: {exc}") from exc

        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected an object")
            if data.get("error"):
                raise OllamaError(f"Ollama error: {data['error']}")
            if data.get("done") is not True:
                raise OllamaError("Ollama did not return a completed answer.")
            if data.get("done_reason") == "length":
                raise OllamaError(
                    "Ollama reached the response token limit before finishing. "
                    "Use OLLAMA_THINK=false or increase the caller's max_tokens."
                )
            content = data["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise OllamaError("Ollama returned an empty answer.")
            if config.response_format == "json":
                json.loads(content)
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an invalid response.") from exc

        return LLMResponse(content=content.strip())

    async def ainvoke(self, prompt: str) -> LLMResponse:
        """Run local inference without blocking FastAPI's event loop."""
        return await asyncio.to_thread(self.invoke, prompt)


def get_llm_client(
    provider: Optional[Literal["ollama"]] = None,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    **kwargs,
) -> OllamaClient:
    """Create the local Qwen client used by all application AI features."""
    return LLMClientConfig(
        provider=provider or "ollama",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    ).create_client()


def get_configured_client(config: Optional[LLMClientConfig] = None) -> OllamaClient:
    """Create a client using explicit configuration or current local defaults."""
    return (config or LLMClientConfig()).create_client()
