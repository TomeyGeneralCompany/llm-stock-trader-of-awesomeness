"""
Daily Decision Improvement API
Decision improvement suggestions for a specific date
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.client import OllamaError, get_llm_client
from app.utils.backtest_logger import BacktestLogger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class DailyFeedbackRequest(BaseModel):
    feedback: str
    date: str  # YYYY-MM-DD
    symbol: str = None  # Optional symbol, if not provided will search all symbols
    run_id: Optional[str] = None  # Exact ownership for staged runs


class DailyImprovementResponse(BaseModel):
    analysis: str
    suggestions: List[str]


@router.post("/daily-feedback", response_model=DailyImprovementResponse)
async def analyze_daily_decision(
    request: DailyFeedbackRequest,
    db_path: str = Query(None, description="Database path"),
) -> DailyImprovementResponse:
    """
    Analyze decisions on a specific date and suggest improvements
    Uses the same data access pattern as the working backtest_analysis API.
    """
    try:
        # 1. Use consistent path across the application
        if not db_path:
            db_path = "backend/data/backtest_logs.db"

        print(f"🔍 Analysis date: {request.date}")
        print(f"📝 User feedback: {request.feedback}")
        print(f"🗄️ Database path: {db_path}")

        if not Path(db_path).exists():
            print(f"❌ Database file does not exist: {db_path}")
            raise HTTPException(
                status_code=404, detail=f"Database not found: {db_path}"
            )

            # 2. Find all sessions that have data for the target date
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # First, get all sessions that have data for this date
        cursor.execute(
            """
            SELECT DISTINCT session_id, symbol FROM daily_analysis_logs 
            WHERE date = ? AND (? IS NULL OR session_id = ?)
            ORDER BY session_id DESC
        """,
            (request.date, request.run_id, request.run_id),
        )
        date_sessions = cursor.fetchall()

        if not date_sessions:
            conn.close()
            raise HTTPException(
                status_code=404, detail=f"Not found {request.date} any trading data"
            )

        print(f"📊 for this date session and stock: {date_sessions}")

        # Determine which session to use
        target_session = None
        target_symbol = None

        if request.symbol:
            # If user specified a symbol, find the session for that symbol
            for session_id, symbol in date_sessions:
                if request.symbol.upper() in symbol or symbol in request.symbol.upper():
                    target_session = session_id
                    target_symbol = symbol
                    break

            if not target_session:
                available_symbols = [symbol for _, symbol in date_sessions]
                conn.close()
                raise HTTPException(
                    status_code=404,
                    detail=f"Stock not found {request.symbol} During {request.date} data.Available stocks: {', '.join(available_symbols)}",
                )
        else:
            # If no symbol specified, prioritize NVDA, TSLA, then others
            priority_symbols = ["NVDA", "TSLA", "AAPL", "MSFT"]

            for priority_symbol in priority_symbols:
                for session_id, symbol in date_sessions:
                    if priority_symbol in symbol:
                        target_session = session_id
                        target_symbol = symbol
                        break
                if target_session:
                    break

            if not target_session:
                # Fallback to first available
                target_session, target_symbol = date_sessions[0]

        print(f"✅ using session: {target_session}")
        print(f"🎯 Query stock: {target_symbol}")
        conn.close()

        # 3. Initialize BacktestLogger and query data
        logger = BacktestLogger(db_path, session_id=target_session)
        logs = logger.query_logs(
            symbol=target_symbol, date_from=request.date, date_to=request.date, limit=1
        )

        if not logs:
            print(
                f"❌ No data found for the specified date: {request.date} (session: {target_session})"
            )
            raise HTTPException(
                status_code=404, detail=f"Not found {request.date} trading data"
            )

        daily_data = logs[0]
        print(
            f"✅ Trading data retrieved successfully: {target_symbol} - {len(daily_data.get('triggered_events', []))} technical events"
        )

        # 5. Read trading strategy content
        strategy_content = load_trading_strategy()

        # 6. using LLM Analyze and generate improvement suggestions
        improvement_response = await generate_daily_improvement_analysis(
            request.feedback, request.date, daily_data, strategy_content
        )

        return improvement_response

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_msg = (
            f"Error processing feedback: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        )
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Remove the old get_daily_trading_data function - will be replaced with simpler logic


def load_trading_strategy() -> str:
    """
    Read trading strategy file contents
    """
    try:
        # Build strategy file path - Correct path calculation
        # Current file: backend/app/api/v1/endpoints/daily_feedback.py
        # Target file: backend/app/llm/strategies/prompt/traditional_strategy.md
        current_file = Path(__file__)  # daily_feedback.py
        app_dir = current_file.parent.parent.parent.parent  # Reach backend/app/
        strategy_path = (
            app_dir / "llm" / "strategies" / "prompt" / "traditional_strategy.md"
        )

        print(f"📋 Read strategy file: {strategy_path}")

        if not strategy_path.exists():
            print(f"⚠️ Strategy file does not exist: {strategy_path}")
            print(f"🔍 Path checked: {strategy_path.absolute()}")
            return "Strategy file not found"

        with open(strategy_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"✅ Strategy file read successfully, Length: {len(content)} characters")
        return content

    except Exception as e:
        print(f"❌ Error reading strategy file: {e}")
        return f"Failed to read strategy file: {str(e)}"


async def generate_daily_improvement_analysis(
    feedback: str, target_date: str, daily_data: Dict[str, Any], strategy_content: str
) -> DailyImprovementResponse:
    """
    Generate daily decision improvement analysis
    """
    try:
        # Prepare data summary
        triggered_events = daily_data.get("triggered_events", [])
        llm_decision = daily_data.get("llm_decision", {})
        symbol = daily_data.get("symbol", "Unknown")
        price = daily_data.get("price", 0)

        # Build LLM Prompt
        context = f"""You are an AI trading strategy discussion partner. The user has feedback about the decision on {target_date}. Analyze the decision and suggest improvements to the strategy file. Write the entire response in English and use the exact section headings below.

=== User’s thoughts ===
{feedback}

=== Conditions on that day ===
{target_date} - {symbol} ${price:.2f}

=== My decision logic at the time ===
{llm_decision.get("decision_type", "N/A")}: {llm_decision.get("reasoning", "N/A")}

=== Current trading strategy file contents ===
{strategy_content}

Analyze in detail and provide specific, actionable suggestions for editing the strategy file!

## My assessment
[First explain why the current strategy made this decision, Then assess the merits of the user’s suggestion, Approximately 2-3 paragraphs]

## Strategy file modification suggestions
Please provide 3 specific modification suggestions, Each suggestion should include: 
- Modification location/section
- Specific text of the new rule
- Actual parameters or conditions

Use the following format: 
1. [Modification title]: [Specify which section of the strategy file should be added to/and which specific rules should be changed, including parameters, conditions, and complete logic, At least 2-3 lines of detailed description]

2. [Modification title]: [Specify which section of the strategy file should be added to/and which specific rules should be changed, including parameters, conditions, and complete logic, At least 2-3 lines of detailed description]

3. [Modification title]: [Specify which section of the strategy file should be added to/and which specific rules should be changed, including parameters, conditions, and complete logic, At least 2-3 lines of detailed description]

## Reasons for modification
[Briefly explain why these changes are needed, and the expected improvements]
"""

        # Get LLM response
        llm_client = get_llm_client(temperature=0.7, max_tokens=1500)
        response = await llm_client.ainvoke(context)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse response
        analysis_parts = parse_llm_response(response_text)

        return DailyImprovementResponse(
            analysis=analysis_parts.get("analysis", "Generating analysis..."),
            suggestions=analysis_parts.get("suggestions", ["Please try again later"]),
        )

    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        print(f"❌ LLM Analysis error: {e}")
        # Return fallback response
        return DailyImprovementResponse(
            analysis=f"Based on your feedback, “{feedback}”, we are analyzing the reasoning behind the decision on {target_date}.",
            suggestions=[
                "Review the combined use of technical indicators",
                "Evaluate market trend assessment accuracy",
                "Optimize risk control parameters",
            ],
            strategy_review="The strategy needs ongoing optimization as market conditions change.",
        )


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM Response text
    """
    try:
        print(f"🔍 Original LLM response:\n{response_text}\n")

        parts = {"analysis": "", "suggestions": [], "strategy_review": ""}

        # Simple text splitting and parsing
        sections = response_text.split("##")
        print(f"📝 After splitting sections Count: {len(sections)}")

        for i, section in enumerate(sections):
            section = section.strip()
            print(f"Section {i}: {section[:100]}...")

            if "My assessment" in section:
                # Remove heading and retain content
                content = section.replace("My assessment", "", 1).strip()
                # Remove possible colons
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["analysis"] = content
                print(f"✅ Found“My assessment”: {content[:50]}...")
            elif "Decision analysis" in section:  # Maintain backward compatibility
                content = section.replace("Decision analysis", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["analysis"] = content
            elif "Strategy file modification suggestions" in section:
                suggestions_text = section.replace("Strategy file modification suggestions", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                print(f"✅ Found“Strategy file modification suggestions”: {suggestions_text[:100]}...")

                # Smarter suggestion extraction - Retain complete content
                suggestions = []
                lines = suggestions_text.split("\n")
                current_suggestion = ""

                for line in lines:
                    line = line.strip()
                    # Check whether a new suggestion begins
                    if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                        # If a previous suggestion exists, save it first
                        if current_suggestion:
                            suggestions.append(current_suggestion.strip())
                        # Start a new suggestion, Remove numbering
                        current_suggestion = line[2:].strip()
                    elif line and current_suggestion:
                        # Continue the current suggestion
                        current_suggestion += "\n" + line
                    elif not current_suggestion and line:
                        # Handle unnumbered suggestion lines
                        current_suggestion = line

                # Add the last suggestion
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())

                print(f"📋 Number of complete suggestions extracted: {len(suggestions)}")
                for i, suggestion in enumerate(suggestions):
                    print(f"Suggestions {i + 1}: {suggestion[:50]}...")

                parts["suggestions"] = suggestions
            elif "Some suggestions" in section:  # Maintain backward compatibility
                suggestions_text = section.replace("Some suggestions", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                # Extract list items
                suggestions = []
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                        or line.startswith("5.")
                    ):
                        suggestions.append(line[2:].strip())
                parts["suggestions"] = suggestions
            elif "Improvement suggestions" in section:  # Maintain backward compatibility
                suggestions_text = section.replace("Improvement suggestions", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                suggestions = []
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                        or line.startswith("5.")
                    ):
                        suggestions.append(line[2:].strip())
                parts["suggestions"] = suggestions
            elif "Reasons for modification" in section:
                content = section.replace("Reasons for modification", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content
            elif "Strategy optimization ideas" in section:  # Maintain backward compatibility
                content = section.replace("Strategy optimization ideas", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content
            elif "Strategy review" in section:  # Maintain backward compatibility
                content = section.replace("Strategy review", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content

        print(
            f"📊 Final parsed result: analysis={bool(parts['analysis'])}, suggestions={len(parts['suggestions'])}, strategy_review={bool(parts['strategy_review'])}"
        )
        return parts

    except Exception as e:
        print(f"❌ Response parsing error: {e}")
        return {
            "analysis": response_text[:200] + "...",
            "suggestions": ["Please try again later"],
            "strategy_review": "Review analysis in progress...",
        }
