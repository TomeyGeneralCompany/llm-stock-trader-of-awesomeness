"""One-question GitHub Actions probe using the application's current local AI."""

import html
import json
import math
import os
import platform
import time
from pathlib import Path

import pandas as pd

from app.backtesting.gates import Stage, frame_records, json_value, validate_output
from app.config import settings
from app.llm.client import get_llm_client
from app.llm.strategies.base import StrategyConfig
from app.llm.strategies.llm_strategy import LLMSmartStrategy
from app.stage_resources import stage_resources


def sample_market() -> pd.DataFrame:
    """Build explicitly synthetic OHLCV data for a single decision review."""
    rows = []
    for day in range(60):
        opening = 100 + day * 0.4 + math.sin(day / 4) * 2
        closing = opening + math.sin(day) * 0.8
        rows.append(
            {
                "open": opening,
                "high": max(opening, closing) + 1.5,
                "low": min(opening, closing) - 1.5,
                "close": closing,
                "volume": 1000000 + day * 10000,
            }
        )
    return pd.DataFrame(rows, index=pd.date_range("2025-01-01", periods=60))


def run_probe(mode: str, prompt: str) -> dict:
    """Ask one question or request one decision with the real strategy prompt."""
    if mode not in ("model-question", "trader-sample"):
        raise ValueError("Choose model-question or trader-sample.")
    if not prompt.strip():
        raise ValueError("Enter a question before running the workflow.")
    started = time.monotonic()
    result = {
        "status": "completed",
        "mode": mode,
        "prompt": prompt,
        "model": settings.OLLAMA_MODEL,
        "commit": os.getenv("GITHUB_SHA", "local"),
        "runner": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    if mode == "model-question":
        result["data_source"] = "No market dataset supplied; free-form model question."
        result["answer"] = (
            get_llm_client(max_tokens=1500)
            .invoke(
                "Answer the following question in English. You are the local Qwen model "
                "used by this trading application. This free-form test includes no live "
                "market data.\n\nQuestion:\n" + prompt
            )
            .content
        )
    else:
        result["data_source"] = (
            "Synthetic SAMPLE instrument; 60 illustrative daily observations, not live market prices."
        )
        data = sample_market()
        strategy = LLMSmartStrategy(
            StrategyConfig(
                name="GitHub sample review",
                description="One decision, no order execution",
                parameters={"enable_logging": False},
            )
        )
        strategy.set_current_symbol("SAMPLE")
        with stage_resources(Stage.PREPARE.value):
            strategy.stock_characteristics = strategy._analyze_stock_characteristics(
                data
            )
            strategy._adjust_technical_parameters()
            prepared = strategy._calculate_all_indicators(data)
        with stage_resources(Stage.ANALYZE.value):
            strategy.trend_analyzer.use_snapshot(
                "SAMPLE",
                [
                    {"date": date.strftime("%Y-%m-%d"), **row.to_dict()}
                    for date, row in data.iterrows()
                ],
            )
            enhanced = strategy.trend_analyzer.analyze_with_llm_optimization(
                "SAMPLE", data.index[-1].strftime("%Y-%m-%d")
            )
            strategy.current_enhanced_analysis = enhanced
            strategy.strategy_prompt += (
                "\n\n## Owner's question about this sample\n"
                + prompt
                + "\nAddress the question in your English reasoning and expected_outcome. "
                "Keep the required JSON decision format."
            )
            events = [
                {
                    "event_type": "MANUAL_REVIEW",
                    "severity": "medium",
                    "description": "The owner requested a review of this synthetic sample.",
                }
            ]
            decision = strategy._make_llm_decision(
                prepared, data.index[-1], events, enhanced.original_result
            )
            if decision is None:
                raise RuntimeError(
                    "The trader did not produce a decision; inspect execution.log."
                )
            # Apply the application's actual decision-output gate to this probe.
            validate_output(
                Stage.ANALYZE,
                json_value(
                    {
                        "rows_processed": len(data),
                        "timestamps": [date.isoformat() for date in data.index],
                        "signals": [],
                        "decisions": strategy.decision_log,
                    }
                ),
                {"rows": frame_records(data)},
            )
        result["decision"] = decision
        result["answer"] = json.dumps(decision, indent=2, ensure_ascii=False)
        result["market_data"] = frame_records(data)
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return result


def write_response(
    result: dict, directory: Path, summary_path: str | None = None
) -> None:
    """Save a transcript and a readable summary without evaluating model text."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "response.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text = (
        "# Ask the Trader\n\n"
        f"**Status:** {html.escape(result['status'])}  \n"
        f"**Mode:** {html.escape(result['mode'])}  \n"
        f"**Model:** {html.escape(result['model'])}  \n"
        f"**Commit:** `{html.escape(result['commit'])}`\n\n"
        f"{html.escape(result.get('data_source', ''))}\n\n"
        "This run uses GitHub's runner. It is a response probe, not a laptop benchmark or a full backtest.\n\n"
        "## Your question\n\n<pre>" + html.escape(result["prompt"]) + "</pre>\n\n"
        "## Response\n\n<pre>"
        + html.escape(result.get("answer", result.get("error", "")))
        + "</pre>\n"
    )
    (directory / "response.md").write_text(text, encoding="utf-8")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as output:
            output.write(text)


def main() -> int:
    mode = os.getenv("TRADER_TEST_MODE", "trader-sample")
    prompt = os.getenv(
        "TRADER_TEST_PROMPT", "Explain your decision and what evidence would change it."
    )
    try:
        result = run_probe(mode, prompt)
    except Exception as exc:
        result = {
            "status": "failed",
            "mode": mode,
            "prompt": prompt,
            "model": settings.OLLAMA_MODEL,
            "commit": os.getenv("GITHUB_SHA", "local"),
            "error": str(exc),
        }
    write_response(
        result,
        Path(os.getenv("TRADER_TEST_OUTPUT", "trader-response")),
        os.getenv("GITHUB_STEP_SUMMARY"),
    )
    print("Response transcript saved in trader-response. Status:", result["status"])
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
