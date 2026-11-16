"""Verify that every application AI feature reaches the local model client."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from app.api.v1.endpoints.backtest_analysis import (
    DayAnalysisData,
    generate_retrospective_analysis,
)
from app.api.v1.endpoints.daily_feedback import generate_daily_improvement_analysis
from app.llm.client import OllamaError
from app.llm.strategies.base import StrategyConfig
from app.llm.strategies.llm_strategy import LLMSmartStrategy
from fastapi import HTTPException


def test_trading_feedback_and_review_use_local_qwen(monkeypatch):
    requests = []
    original = httpx.Client

    def handler(request):
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "qwen3:4b"
        requests.append(body)
        if body.get("format") == "json":
            content = json.dumps(
                {
                    "action": "HOLD",
                    "confidence": 0.8,
                    "reasoning": "Wait for confirmation.",
                    "risk_level": "medium",
                    "expected_outcome": "Continue observing the trend.",
                }
            )
        elif "Strategy file modification suggestions" in body["messages"][0]["content"]:
            content = (
                "## My assessment\nThe strategy waited for confirmation.\n"
                "## Strategy file modification suggestions\n"
                "1. Entry confirmation: Clarify the existing volume condition.\n"
                "2. Exit confirmation: Clarify the moving-average condition.\n"
                "3. Review: Compare the two conditions.\n"
                "## Reasons for modification\nMake the rules clearer.\n"
            )
        else:
            content = "The decision followed the existing trend rules."
        return httpx.Response(200, json={"done": True, "message": {"content": content}})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    strategy = LLMSmartStrategy(
        StrategyConfig(
            name="local-test", description="Test", parameters={"enable_logging": False}
        )
    )
    strategy.stock_characteristics = {
        "volatility": 0.2,
        "trend_consistency": 0.7,
        "macd_effectiveness": 0.6,
    }
    index = pd.date_range("2025-01-01", periods=60)
    data = pd.DataFrame(
        {
            "open": [100 + i for i in range(60)],
            "high": [102 + i for i in range(60)],
            "low": [99 + i for i in range(60)],
            "close": [101 + i for i in range(60)],
            "volume": [1000000] * 60,
        },
        index=index,
    )
    decision = strategy._make_llm_decision(
        data,
        index[-1],
        [
            {
                "event_type": "MACD_GOLDEN_CROSS",
                "description": "Bullish crossover",
                "severity": "medium",
            }
        ],
        SimpleNamespace(dominant_trend="uptrend", complexity_score=0.3),
    )
    assert decision is not None
    assert decision["action"] == "HOLD"
    assert strategy.decision_log[-1]["reasoning"] == "Wait for confirmation."

    feedback = asyncio.run(
        generate_daily_improvement_analysis(
            "Explain the decision",
            "2025-03-01",
            {"symbol": "AAPL", "price": 159, "llm_decision": {"decision_type": "HOLD"}},
            strategy.strategy_prompt,
        )
    )
    assert feedback.analysis == "The strategy waited for confirmation."
    assert len(feedback.suggestions) == 3
    review = asyncio.run(
        generate_retrospective_analysis(
            DayAnalysisData(date="2025-03-01", symbol="AAPL", price=159)
        )
    )
    assert review.llm_commentary == "The decision followed the existing trend rules."
    assert len(requests) == 3
    assert requests[0]["format"] == "json"
    assert all("format" not in request for request in requests[1:])


def test_local_model_failure_is_not_a_no_trade_decision(monkeypatch):
    strategy = LLMSmartStrategy(
        StrategyConfig(
            name="local-test", description="Test", parameters={"enable_logging": False}
        )
    )
    data = pd.DataFrame({"close": [100]}, index=pd.date_range("2025-01-01", periods=1))
    monkeypatch.setattr(
        strategy.enhanced_analyzer, "analyze_comprehensive_context", lambda *a, **k: {}
    )
    monkeypatch.setattr(strategy, "_build_decision_prompt", lambda *a, **k: "Review")

    def unavailable(*args):
        raise OllamaError("Start Ollama with 'ollama serve'.")

    monkeypatch.setattr(strategy.llm_client, "invoke", unavailable)
    with pytest.raises(OllamaError, match="ollama serve"):
        strategy._make_llm_decision(data, data.index[0], [], None)
    assert strategy.decision_log == []


def test_feedback_surfaces_missing_model(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(404)), **kwargs
        ),
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            generate_daily_improvement_analysis(
                "Review", "2025-03-01", {"symbol": "AAPL", "price": 100}, "Rules"
            )
        )
    assert error.value.status_code == 503
    assert "ollama pull qwen3:4b" in error.value.detail
