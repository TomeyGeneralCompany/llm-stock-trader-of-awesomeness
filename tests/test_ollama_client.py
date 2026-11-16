"""Exercise the Ollama API contract without downloading or running a model."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from app.config import Settings
from app.llm import client as local_ai


@pytest.fixture
def mock_ollama(monkeypatch):
    """Use a real HTTPX client with an in-memory Ollama transport."""
    original = httpx.Client

    def install(handler):
        def create(**kwargs):
            return original(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(local_ai.httpx, "Client", create)

    return install


def answer(content="Local answer", **extra):
    return httpx.Response(
        200,
        json={
            "done": True,
            "done_reason": "stop",
            "message": {"content": content, "thinking": "Separate thinking"},
            **extra,
        },
    )


def test_defaults_use_qwen_without_cloud_credentials(monkeypatch):
    for key in ("AZURE_OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "old-unused-key")
    config = local_ai.LLMClientConfig()
    assert config.provider == "ollama"
    assert config.model == "qwen3:4b"
    assert config.num_thread == 4
    assert config.num_ctx == 8192
    assert config.timeout_seconds == 0
    assert not config.think
    with pytest.raises(ValueError, match="Only the local Ollama"):
        local_ai.get_llm_client(provider="azure")


def test_env_settings_and_old_env_file_are_compatible(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "OLLAMA_MODEL=qwen3:custom\nOLLAMA_NUM_THREAD=3\n"
        "OLLAMA_NUM_CTX=4096\nOLLAMA_THINK=true\n"
        "OLLAMA_TIMEOUT_SECONDS=900\nGOOGLE_API_KEY=unused\n"
    )
    config = Settings(_env_file=env)
    assert config.OLLAMA_MODEL == "qwen3:custom"
    assert config.OLLAMA_NUM_THREAD == 3
    assert config.OLLAMA_NUM_CTX == 4096
    assert config.OLLAMA_THINK
    assert config.OLLAMA_TIMEOUT_SECONDS == 900


def test_native_json_request_and_final_answer(mock_ollama):
    result = {"action": "HOLD", "confidence": 0.7, "reasoning": "Wait."}
    requests = []

    def handler(request):
        requests.append(request)
        return answer(json.dumps(result))

    mock_ollama(handler)
    client = local_ai.get_llm_client(response_format="json")
    response = client.invoke("Return a JSON decision in English.")
    assert json.loads(response.content) == result
    request = requests[0]
    assert str(request.url) == "http://127.0.0.1:11434/api/chat"
    assert "authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload == {
        "model": "qwen3:4b",
        "messages": [{"role": "user", "content": "Return a JSON decision in English."}],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 4000,
            "num_ctx": 8192,
            "num_thread": 4,
        },
    }
    assert "Separate thinking" not in response.content


def test_configured_client_prose_and_runtime_overrides(mock_ollama):
    def handler(request):
        assert str(request.url) == "http://localhost:11435/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3:custom"
        assert payload["think"] is True
        assert payload["options"]["num_predict"] == 1200
        assert payload["options"]["num_thread"] == 2
        assert payload["options"]["num_ctx"] == 4096
        assert "format" not in payload
        assert request.extensions["timeout"]["read"] == 900
        return answer("  English strategy review.  ")

    mock_ollama(handler)
    config = local_ai.LLMClientConfig(
        model="qwen3:custom",
        base_url="http://localhost:11435/",
        think=True,
        max_tokens=1200,
        num_thread=2,
        num_ctx=4096,
        timeout_seconds=900,
    )
    assert (
        local_ai.get_configured_client(config).invoke("Review this decision").content
        == "English strategy review."
    )


def test_slow_cpu_has_no_read_deadline(mock_ollama):
    def handler(request):
        assert request.extensions["timeout"] == {
            "connect": 5.0,
            "read": None,
            "write": 30.0,
            "pool": 5.0,
        }
        return answer()

    mock_ollama(handler)
    assert local_ai.get_llm_client().invoke("Review").content == "Local answer"


@pytest.mark.parametrize(
    "response, message",
    [
        (httpx.Response(404, json={"error": "model missing"}), "ollama pull qwen3:4b"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, text="invalid"), "invalid response"),
        (httpx.Response(200, json=[]), "invalid response"),
        (httpx.Response(200, json={"error": "model failed"}), "model failed"),
        (answer(done=False), "completed answer"),
        (answer(done_reason="length"), "response token limit"),
        (answer(""), "empty answer"),
        (httpx.Response(200, json={"done": True}), "invalid response"),
    ],
)
def test_failures_are_actionable(mock_ollama, response, message):
    mock_ollama(lambda request: response)
    with pytest.raises(local_ai.OllamaError, match=message):
        local_ai.get_llm_client().invoke("Analyze")


@pytest.mark.parametrize(
    "error, message",
    [
        (httpx.ConnectError, "ollama serve"),
        (httpx.ReadTimeout, "OLLAMA_TIMEOUT_SECONDS"),
    ],
)
def test_transport_errors(mock_ollama, error, message):
    def handler(request):
        raise error("test failure", request=request)

    mock_ollama(handler)
    with pytest.raises(local_ai.OllamaError, match=message):
        local_ai.get_llm_client().invoke("Analyze")


def test_json_mode_rejects_non_json_answer(mock_ollama):
    mock_ollama(lambda request: answer("This is prose."))
    with pytest.raises(local_ai.OllamaError, match="invalid response"):
        local_ai.get_llm_client(response_format="json").invoke("Return JSON")


def test_all_clients_share_one_inference_slot(mock_ollama):
    active = 0
    peak = 0
    guard = threading.Lock()
    release = threading.Event()
    entered = threading.Event()

    def handler(request):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        entered.set()
        release.wait(timeout=5)
        with guard:
            active -= 1
        return answer()

    mock_ollama(handler)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(local_ai.get_llm_client().invoke, "Review") for _ in range(3)
        ]
        assert entered.wait(timeout=5)
        release.set()
        assert all(job.result(timeout=5).content == "Local answer" for job in jobs)
    assert peak == 1


def test_async_inference_leaves_the_event_loop_available(mock_ollama):
    entered = threading.Event()
    release = threading.Event()

    def handler(request):
        entered.set()
        assert release.wait(timeout=5)
        return answer()

    mock_ollama(handler)

    async def run():
        task = asyncio.create_task(local_ai.get_llm_client().ainvoke("Review"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            # This coroutine can proceed while the model request is still waiting.
            assert not task.done()
            release.set()
            assert (await task).content == "Local answer"
        finally:
            release.set()

    asyncio.run(run())
