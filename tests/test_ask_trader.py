"""Verify the GitHub response probe uses the app client and handles real errors."""

import json

import httpx
import pytest
from app import ask_trader
from decision_fixtures import explained


def use_ollama(monkeypatch, content):
    original = httpx.Client
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        answer = content
        if body.get("format") == "json":
            answer = json.dumps(
                explained(json.loads(content), body["messages"][0]["content"])
            )
        return httpx.Response(200, json={"done": True, "message": {"content": answer}})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    return requests


def test_free_form_question_uses_the_current_local_client(monkeypatch):
    requests = use_ollama(monkeypatch, "Moving averages summarize recent prices.")
    result = ask_trader.run_probe("model-question", "Explain moving averages.")
    assert result["answer"] == "Moving averages summarize recent prices."
    assert len(requests) == 1 and requests[0]["model"] == "qwen3:4b"
    assert "Explain moving averages." in requests[0]["messages"][0]["content"]
    assert "format" not in requests[0]


def test_sample_decision_uses_strategy_prompt_and_output_gate(monkeypatch):
    decision = {
        "action": "HOLD",
        "confidence": 0.8,
        "risk_level": "medium",
        "reasoning": "Wait for confirmation.",
        "expected_outcome": "Continue observing.",
    }
    requests = use_ollama(monkeypatch, json.dumps(decision))
    result = ask_trader.run_probe(
        "trader-sample", "What evidence would change your decision?"
    )
    assert {key: result["decision"][key] for key in decision} == decision
    assert result["decision"]["explanation"]["forecast"]["assumptions"]
    assert len(requests) == 1 and requests[0]["format"] == "json"
    assert "Traditional decision principles" in requests[0]["messages"][0]["content"]
    assert (
        "What evidence would change your decision?"
        in requests[0]["messages"][0]["content"]
    )
    assert len(result["market_data"]) == 60
    assert "Synthetic SAMPLE" in result["data_source"]


def test_bad_model_decision_cannot_be_reported_as_success(monkeypatch):
    use_ollama(monkeypatch, json.dumps({"action": "INVENTED", "confidence": 1}))
    with pytest.raises(Exception, match="model decision"):
        ask_trader.run_probe("trader-sample", "Explain")


def test_summary_escapes_prompt_and_model_text(tmp_path):
    result = {
        "status": "completed",
        "mode": "model-question",
        "model": "qwen3:4b",
        "commit": "test",
        "prompt": "<script>example</script>",
        "answer": "<b>literal text</b>",
    }
    summary = tmp_path / "summary.md"
    ask_trader.write_response(result, tmp_path / "output", str(summary))
    text = summary.read_text()
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "&lt;b&gt;literal text&lt;/b&gt;" in text
    assert json.loads((tmp_path / "output/response.json").read_text()) == result


def test_probe_failure_writes_diagnostics_and_returns_nonzero(monkeypatch, tmp_path):
    def unavailable(*args):
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(ask_trader, "run_probe", unavailable)
    monkeypatch.setenv("TRADER_TEST_OUTPUT", str(tmp_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "test-summary.md"))
    assert ask_trader.main() == 1
    result = json.loads((tmp_path / "response.json").read_text())
    assert result["status"] == "failed" and "Ollama unavailable" in result["error"]


@pytest.mark.parametrize(
    "mode,prompt", [("invalid", "Question"), ("model-question", "  ")]
)
def test_invalid_inputs_do_not_start_inference(mode, prompt):
    with pytest.raises(ValueError):
        ask_trader.run_probe(mode, prompt)
