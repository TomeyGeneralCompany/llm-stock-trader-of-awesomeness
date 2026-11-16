"""Exercise retrieval boundaries and the real placeholder regression."""

import json
import sqlite3

import httpx
import pytest
from app import ask_trader
from decision_fixtures import explained
from app.backtesting.gates import GateError, Stage, validate_output
from app.llm.client import OllamaError, get_llm_client
from app.llm.decisions import DecisionError
from app.llm.memory import DecisionMemory, digest

PLACEHOLDER = {
    "action": "HOLD",
    "confidence": 0.0,
    "risk_level": "medium",
    "reasoning": "Detailed decision reasoning",
    "expected_outcome": "Description of expected outcome",
}
VALID = {
    **PLACEHOLDER,
    "confidence": 0.7,
    "reasoning": "The synthetic close is above MA20; wait for confirmation.",
    "expected_outcome": "An established uptrend with bullish MACD would favor entry.",
}


def assemble(memory, date="2025-02-01", **kwargs):
    return memory.assemble(
        asof=date,
        instructions="Policy",
        working={"cash": 100},
        evidence="Current price is 125. Volume 1000.",
        query="price",
        budget=2500,
        **kwargs,
    )


def test_file_chunks_preserve_original_offsets_overlap_and_unicode(
    tmp_path, monkeypatch
):
    file = tmp_path / "rules.md"
    original = "alpha β世界\n" * 500
    file.write_text(original)
    memory = DecisionMemory(path=tmp_path / "memory.db")
    monkeypatch.setattr(
        type(file),
        "read_text",
        lambda *a, **k: pytest.fail("File chunking read the entire file at once"),
    )
    memory.archive_file(file)
    assert memory.original("file:rules.md") == original
    with memory.connect() as db:
        chunks = db.execute(
            "SELECT start,end,body,sha FROM memory_chunks ORDER BY ordinal"
        ).fetchall()
    previous_end = 0
    for start, end, body, sha in chunks:
        assert original[start:end] == body and digest(body) == sha
        assert start <= previous_end
        previous_end = end
    assert previous_end == len(original)
    with pytest.raises(ValueError, match="immutable"):
        memory.archive("file:rules.md", "changed")


def test_tiers_exclude_future_and_other_runs_and_recover_originals(tmp_path):
    path = tmp_path / "memory.db"
    memory = DecisionMemory("run-one", path)
    memory.archive("analysis:future", "FUTURE_SECRET price", "2026-01-01")
    memory.remember("2025-01-01", VALID)
    memory.remember("2026-01-01", {**VALID, "reasoning": "FUTURE_EPISODE"})
    other = DecisionMemory("other", path)
    other.archive("analysis:other", "OTHER_SECRET price", "2025-01-01")
    prompt, receipt = assemble(memory)
    assert "FUTURE_" not in prompt and "OTHER_SECRET" not in prompt
    assert "bullish MACD" in prompt and '"cash": 100' in prompt
    assert len(prompt.encode()) <= receipt["budget_bytes"]
    assert any(r["source"].startswith("decision:") for r in receipt["references"])
    restarted = DecisionMemory("run-one", path)
    assert restarted.original("prompt:2025-02-01") == prompt
    assert digest(prompt) == receipt["prompt_sha256"]


def test_mandatory_state_cannot_be_silently_truncated(tmp_path):
    memory = DecisionMemory(path=tmp_path / "memory.db")
    with pytest.raises(ValueError, match="Required policy"):
        memory.assemble(
            asof="2025-01-01",
            instructions="policy" * 500,
            working={},
            evidence="evidence",
            query="question",
            budget=1000,
        )


def test_tampered_evidence_is_rejected(tmp_path):
    memory = DecisionMemory(path=tmp_path / "memory.db")
    memory.archive("analysis:old", "price 100", "2025-01-01")
    with sqlite3.connect(memory.path) as db:
        db.execute("UPDATE memory_chunks SET body='tampered'")
    with pytest.raises(ValueError, match="integrity"):
        assemble(memory)


def test_client_rejects_over_budget_before_http(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Over-budget prompt reached HTTP")

    monkeypatch.setattr(httpx, "Client", forbidden)
    with pytest.raises(OllamaError, match="context budget"):
        get_llm_client().invoke("x" * 9000)


@pytest.mark.parametrize("corrected", [True, False])
def test_real_sample_placeholder_gets_one_correction_then_validates(
    monkeypatch, corrected
):
    original = httpx.Client
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        decision = VALID if corrected and len(requests) == 2 else PLACEHOLDER
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {
                    "content": json.dumps(
                        explained(decision, requests[-1]["messages"][0]["content"])
                    )
                },
            },
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    if corrected:
        result = ask_trader.run_probe(
            "trader-sample", "What would change this decision?"
        )
        assert {key: result["decision"][key] for key in VALID} == VALID
        assert result["memory"]["attempts"] == 2
        assert result["memory"]["final_prompt_sha256"] == digest(
            requests[-1]["messages"][0]["content"]
        )
        assert result["memory"]["references"]
    else:
        with pytest.raises(DecisionError, match="after correction"):
            ask_trader.run_probe("trader-sample", "Explain")
    assert len(requests) == 2
    assert "Correction:" in requests[1]["messages"][0]["content"]
    assert all(
        len(r["messages"][0]["content"].encode()) + r["options"]["num_predict"] + 256
        <= r["options"]["num_ctx"]
        for r in requests
    )


def test_sealed_gate_independently_rejects_placeholder():
    date = "2025-01-01T00:00:00"
    with pytest.raises(GateError, match="placeholder"):
        validate_output(
            Stage.ANALYZE,
            {
                "rows_processed": 1,
                "timestamps": [date],
                "signals": [],
                "decisions": [{"date": date, "decision": PLACEHOLDER}],
            },
            {"rows": [{"timestamp": date}]},
        )


def test_large_warm_decision_uses_labelled_excerpt_without_losing_original(tmp_path):
    memory = DecisionMemory("test", tmp_path / "memory.db")
    long = {**VALID, "reasoning": "Past evidence " * 1500}
    memory.remember("2025-01-01", long)
    prompt, receipt = assemble(memory)
    assert "excerpt of prior decision" in prompt
    ref = next(r for r in receipt["references"] if r["source"].startswith("decision:"))
    with memory.connect() as db:
        original = db.execute(
            "SELECT body FROM memory_episodes WHERE run='test'"
        ).fetchone()[0]
    assert json.loads(original) == long
    assert ref["original_sha256"] == digest(original)
    assert ref["sha256"] == digest(original[ref["start"] : ref["end"]])
    assert len(prompt.encode()) <= 2500


def test_decision_working_state_cannot_read_later_rows(monkeypatch):
    import pandas as pd
    from app.llm.strategies.base import StrategyConfig
    from app.llm.strategies.llm_strategy import LLMSmartStrategy

    strategy = LLMSmartStrategy(
        StrategyConfig(
            name="date-check", description="Test", parameters={"enable_logging": False}
        )
    )
    monkeypatch.setattr(
        strategy.enhanced_analyzer, "analyze_comprehensive_context", lambda *a, **k: {}
    )
    monkeypatch.setattr(strategy, "_build_decision_prompt", lambda *a, **k: "Evidence")
    from app.llm.client import LLMResponse

    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return LLMResponse(json.dumps(explained(VALID, prompt)))

    monkeypatch.setattr(strategy.llm_client, "invoke", answer)
    data = pd.DataFrame(
        {"close": [100, 99999]}, index=pd.date_range("2025-01-01", periods=2)
    )
    strategy._make_llm_decision(data, data.index[0], [], None)
    assert '"close": 100' in prompts[0]
    assert "99999" not in prompts[0]
