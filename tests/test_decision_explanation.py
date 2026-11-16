"""Preserve forecast freedom while requiring explicit evidence and prerequisites."""

import copy
import json

import pytest
from app import ask_trader
from app.backtesting.gates import GateError, Stage, validate_output
from app.llm.decisions import (
    DecisionError,
    format_decision_reasoning,
    validate_decision,
)
from decision_fixtures import explained


@pytest.fixture
def case():
    context = {
        "asof": "2025-03-01T00:00:00",
        "shares": 0,
        "observation_timeframe": "daily bars through March 1",
        "trend": {"market_phase": "uptrend"},
        "macd_cross": "none",
        "current": {"close": 125.75},
    }
    prompt = (
        "Controller working state (authoritative):\n"
        + json.dumps(context)
        + "\n\nOwner question:\nExplain"
    )
    decision = explained(
        {
            "action": "HOLD",
            "confidence": 0.8,
            "risk_level": "low",
            "reasoning": "Wait below resistance while flat.",
            "expected_outcome": "An upward breakout would support an entry.",
        },
        prompt,
    )
    return decision, context


def test_forecast_can_include_consolidation_inside_uptrend_and_future_sell(case):
    decision, context = case
    decision["explanation"]["interpretation"] = (
        "The broad daily uptrend contains a possible shorter pause; a single quiet bar does not establish consolidation."
    )
    decision["explanation"]["forecast"]["scenario"] = (
        "A bullish crossover may occur in later daily bars; it has not occurred now."
    )
    assert validate_decision(decision, context) == decision
    text = format_decision_reasoning(decision)
    for label in [
        "Observed state",
        "Interpretation",
        "Forecast",
        "Assumptions",
        "Invalidation",
        "Conditional actions",
        "Execute the BUY first",
    ]:
        assert label in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("position", "long"),
        ("market_phase", "consolidation"),
        ("macd_cross", "bullish_cross"),
    ],
)
def test_observed_facts_must_match_controller(case, field, value):
    decision, context = case
    decision["explanation"]["observed_state"][field] = value
    with pytest.raises(DecisionError, match="match controller"):
        validate_decision(decision, context)


@pytest.mark.parametrize(
    "field", ["observed_state", "interpretation", "forecast", "conditional_actions"]
)
def test_answer_alone_cannot_replace_explanation(case, field):
    decision, context = case
    del decision["explanation"][field]
    with pytest.raises(DecisionError):
        validate_decision(decision, context)


def test_current_sell_is_rejected_when_flat_but_allowed_when_long(case):
    decision, context = case
    decision["action"] = "SELL"
    with pytest.raises(DecisionError, match="Current SELL"):
        validate_decision(decision, context)
    context["shares"] = 10
    decision["explanation"]["observed_state"]["position"] = "long"
    assert validate_decision(decision, context)


def test_future_sell_requires_long_prerequisite(case):
    decision, context = case
    decision["explanation"]["conditional_actions"][-1]["required_position"] = "any"
    with pytest.raises(DecisionError, match="Conditional SELL"):
        validate_decision(decision, context)


def test_analysis_gate_independently_validates_explanation(case):
    decision, context = case
    decision["explanation"]["observed_state"]["macd_cross"] = "bullish_cross"
    with pytest.raises(GateError, match="match controller"):
        validate_output(
            Stage.ANALYZE,
            {
                "rows_processed": 1,
                "timestamps": [context["asof"]],
                "signals": [],
                "decisions": [
                    {"date": context["asof"], "context": context, "decision": decision}
                ],
            },
            {"rows": [{"timestamp": context["asof"]}]},
        )


def test_sample_requires_correction_for_missing_explanation(monkeypatch):
    import httpx

    original = httpx.Client
    requests = []
    plain = {
        "action": "HOLD",
        "confidence": 0.8,
        "risk_level": "low",
        "reasoning": "Wait below resistance.",
        "expected_outcome": "Wait for breakout confirmation.",
    }

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        decision = (
            copy.deepcopy(plain)
            if len(requests) == 1
            else explained(plain, body["messages"][0]["content"])
        )
        return httpx.Response(
            200, json={"done": True, "message": {"content": json.dumps(decision)}}
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = ask_trader.run_probe(
        "trader-sample", "Explain the decision and the possible future transitions."
    )
    assert result["memory"]["attempts"] == 2
    assert "requires a structured explanation" in requests[1]["messages"][0]["content"]
    assert (
        result["decision"]["explanation"]["conditional_actions"][-1][
            "required_position"
        ]
        == "long"
    )
    for request in requests:
        assert len(request["messages"][0]["content"].encode()) + 1500 + 256 <= 8192


def test_memory_receipt_requires_its_original_controller_snapshot(case):
    from app.llm.memory import digest

    decision, context = case
    receipt = {
        "working_sha256": digest(
            json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
        )
    }
    output = {
        "rows_processed": 1,
        "timestamps": [context["asof"]],
        "signals": [],
        "decisions": [
            {"date": context["asof"], "memory": receipt, "decision": decision}
        ],
    }
    previous = {"rows": [{"timestamp": context["asof"]}]}
    with pytest.raises(GateError, match="missing its dated controller context"):
        validate_output(Stage.ANALYZE, output, previous)
    output["decisions"][0]["context"] = {**context, "shares": 10}
    with pytest.raises(GateError, match="memory receipt"):
        validate_output(Stage.ANALYZE, output, previous)


def test_github_summary_displays_reasoning_sections_and_escapes_text(case, tmp_path):
    decision, _ = case
    decision["explanation"]["forecast"]["scenario"] = "<script>future scenario</script>"
    ask_trader.write_response(
        {
            "status": "completed",
            "mode": "trader-sample",
            "model": "qwen3:4b",
            "commit": "test",
            "prompt": "Explain",
            "decision": decision,
        },
        tmp_path,
    )
    summary = (tmp_path / "response.md").read_text()
    assert "Observed state" in summary and "Conditional actions" in summary
    assert "&lt;script&gt;future scenario&lt;/script&gt;" in summary
    assert "<script>" not in summary
    assert json.loads((tmp_path / "response.json").read_text())["decision"] == decision
