"""
LLM Strategy Streaming Backtest API - Server-Sent Events (SSE)
Provides real-time progress updates and result streaming
"""

import json
import queue
import threading
from datetime import datetime
from typing import Generator, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ....backtesting.engine import BacktestConfig, BacktestEngine
from ....backtesting.gates import (
    GateCancelled,
    GateError,
    GateStore,
    Stage,
    StageGateController,
    frame_records,
)
from ....llm.strategies.base import StrategyConfig
from ....llm.strategies.llm_strategy import (
    LLMSmartStrategy,  # Switch back to original version
)
from ....utils.stock_data import StockService

router = APIRouter()


def safe_json_serialize(obj):
    """
    Safe JSON serialization function, Handle objects that are not directly serializable
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        # DataFrame Convert to a list of dictionaries
        return obj.to_dict("records") if hasattr(obj, "to_dict") else str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, "to_dict"):
        # If the object has to_dict method, use it
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        # For custom objects, Try serializing their attributes
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    else:
        return str(obj)


def safe_json_dumps(obj):
    """
    Safe JSON dumps, Preprocess objects that are not directly serializable
    """

    def json_serializer(o):
        return safe_json_serialize(o)

    return json.dumps(obj, default=json_serializer, ensure_ascii=False)


@router.get("/llm-backtest-stream")
async def stream_llm_backtest(
    symbol: str = Query(..., description="Stock symbol"),
    period: str = Query("1y", description="Backtest period"),
    max_position_size: float = Query(0.3, gt=0, le=1, description="Maximum position allocation"),
    stop_loss: float = Query(0.05, ge=0, lt=1, description="Stop-loss fraction"),
    take_profit: float = Query(0.1, ge=0, description="Take-profit fraction"),
    initial_capital: float = Query(10000.0, gt=0, description="Starting cash"),
    commission_rate: float = Query(0.001, ge=0, lt=1, description="Commission fraction per fill"),
    min_commission: float = Query(0.0, ge=0, description="Minimum commission per fill"),
    session_id: Optional[str] = Query(
        None,
        min_length=1,
        max_length=120,
        description="Run identifier for stage evidence and analysis logs",
    ),
):
    """
    Streaming LLM strategy backtest - using Server-Sent Events
    Decisions and historical fills share a finite controller-owned account.
    """
    try:

        def generate_backtest_stream() -> Generator[str, None, None]:
            """Generate backtest progress stream"""

            # Create a message queue for communication between threads
            message_queue = queue.Queue()

            def progress_callback(
                day: int,
                total_days: int,
                event_type: str,
                message: str,
                extra_data: dict = None,
            ):
                """Progress callback"""
                progress_data = {
                    "type": "trading_progress",
                    "day": day,
                    "total_days": total_days,
                    "progress": round((day / total_days) * 100, 1),
                    "event_type": event_type,
                    "message": message,
                }

                # If additional data exists (such as P&L Information) , add it to progress data
                if extra_data:
                    progress_data.update(
                        {k: v for k, v in extra_data.items() if k not in progress_data}
                    )

                message_queue.put(progress_data)

            def run_backtest():
                """Run backtest in a separate thread"""
                gates = None
                try:
                    gates = StageGateController(
                        run_id=session_id,
                        request={
                            "symbol": symbol,
                            "period": period,
                            "initial_capital": initial_capital,
                            "commission_rate": commission_rate,
                            "min_commission": min_commission,
                            "max_position_size": max_position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                        },
                        callback=lambda state: message_queue.put(
                            {"type": "stage_gate", "data": state}
                        ),
                    )
                    gates.start(Stage.ACQUIRE)
                    # 1. Get stock data
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "data_loading",
                            "message": f"Fetching {symbol} Stock data...",
                        }
                    )

                    stock_service = StockService()
                    stock_data_list = stock_service.get_market_data(symbol, period)

                    if not stock_data_list or len(stock_data_list) < 30:
                        raise GateError(
                            "Insufficient stock data; at least 30 observations are required."
                        )

                    # Convert to DataFrame, backtest engine Requires DataFrame format
                    stock_data = pd.DataFrame(stock_data_list)
                    # Set date as index, and ensure column names match expectations
                    stock_data["date"] = pd.to_datetime(stock_data["date"])
                    stock_data.set_index("date", inplace=True)
                    # Rename columns to standard format (lowercase, because the strategy and engine expect lowercase column names)
                    stock_data.columns = [
                        str(column).lower() for column in stock_data.columns
                    ]
                    gates.complete(Stage.ACQUIRE, {"rows": frame_records(stock_data)})

                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "data_loaded",
                            "message": f"Successfully retrieved {len(stock_data)} days of data",
                        }
                    )

                    # 2. Initialize strategy
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "strategy_init",
                            "message": "Initialize LLM Strategy...",
                        }
                    )

                    # Create LLM Strategy configuration - using .env fixed model from
                    strategy_config = StrategyConfig(
                        name="LLM Trading Strategy",
                        description="AI-powered trading strategy with real-time analysis",
                        parameters={
                            "initial_capital": initial_capital,
                            "commission_rate": commission_rate,
                            "min_commission": min_commission,
                            "max_position_size": max_position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "progress_callback": progress_callback,
                            # Enable backtest logging
                            "enable_logging": True,
                            "session_id": gates.run_id,
                        },
                    )

                    strategy = LLMSmartStrategy(strategy_config)

                    # 3. Start backtest
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "backtest_start",
                            "message": "Start backtest execution...",
                        }
                    )

                    # Create backtest configuration
                    backtest_config = BacktestConfig(initial_capital=initial_capital)
                    backtest_engine = BacktestEngine(backtest_config)

                    def build_report(backtest_result):
                        # Convert DataFrame Convert to a JSON serializable format
                        stock_data_json = []
                        for row in gates.store.artifact(gates.run_id, Stage.ACQUIRE)[
                            "rows"
                        ]:
                            stock_data_json.append(
                                {
                                    "timestamp": pd.Timestamp(
                                        row["timestamp"]
                                    ).strftime("%Y-%m-%d"),
                                    "open": float(row["open"]),
                                    "high": float(row["high"]),
                                    "low": float(row["low"]),
                                    "close": float(row["close"]),
                                    "volume": int(row["volume"]),
                                }
                            )

                        # Extract from strategy object LLM Decision recording
                        llm_decisions = []
                        for log_entry in gates.store.artifact(
                            gates.run_id, Stage.ANALYZE
                        )["decisions"]:
                            # Handle safely decision, Ensure no operation on None Call .get()
                            decision = log_entry.get("decision", {})
                            if decision is None:
                                decision = {}

                            llm_decisions.append(
                                {
                                    "timestamp": log_entry["date"].isoformat()
                                    if hasattr(log_entry["date"], "isoformat")
                                    else str(log_entry["date"]),
                                    "decision": decision,
                                    "reasoning": log_entry.get("reasoning", ""),
                                    "events": log_entry.get("events", []),
                                    "action": "THINK",  # LLM Reasoning decision, Non-trading signal
                                    "confidence": decision.get("confidence", 0.8),
                                    "price": 0.0,  # The frontend will find the corresponding price by timestamp
                                }
                            )

                        result_data = {
                            "type": "result",
                            "data": {
                                "trades": backtest_result.get("trades", []),
                                "performance": backtest_result.get(
                                    "performance_metrics", {}
                                ),
                                "stock_data": stock_data_json,
                                "signals": backtest_result.get(
                                    "trading_signals", []
                                ),  # Correction: using trading_signals instead of signals
                                "llm_decisions": llm_decisions,
                                "statistics": {
                                    # Use strategy statistics, These are actual trading P&L figures
                                    "total_trades": backtest_result.get(
                                        "strategy_statistics", {}
                                    ).get("total_trades", 0),
                                    "win_rate": backtest_result.get(
                                        "strategy_statistics", {}
                                    ).get("strategy_win_rate", 0)
                                    * 100,  # Convert to percentage
                                    "total_return": backtest_result.get(
                                        "strategy_statistics", {}
                                    ).get("cumulative_trade_return_rate", 0)
                                    * 100,  # Convert to percentage
                                    "max_drawdown": backtest_result.get(
                                        "performance_metrics", {}
                                    ).get("max_drawdown", 0),
                                    "final_value": backtest_result.get(
                                        "performance_metrics", {}
                                    ).get("final_value", 0),
                                    "annual_return": backtest_result.get(
                                        "performance_metrics", {}
                                    ).get("annual_return", 0),
                                    "volatility": backtest_result.get(
                                        "performance_metrics", {}
                                    ).get("volatility", 0),
                                    "num_trades": backtest_result.get(
                                        "strategy_statistics", {}
                                    ).get("total_trades", 0),
                                    # Add strategy-specific realized P&L
                                    "total_realized_pnl": backtest_result.get(
                                        "strategy_statistics", {}
                                    ).get("total_realized_pnl", 0),
                                    "winning_trades": backtest_result.get(
                                        "strategy_statistics", {}
                                    ).get("winning_trades", 0),
                                },
                                "strategy_statistics": backtest_result.get(
                                    "strategy_statistics", {}
                                ),
                            },
                        }

                        result_data["data"]["execution_settings"] = backtest_result["execution_settings"]
                        result_data["data"]["orders"] = backtest_result["orders"]
                        result_data["data"]["run_id"] = gates.run_id
                        return result_data["data"]

                    # Run backtest
                    backtest_result = backtest_engine.run_backtest(
                        stock_data=stock_data,
                        strategy=strategy,
                        initial_cash=initial_capital,
                        transaction_cost=commission_rate,
                        min_commission=min_commission,
                        max_position_size=max_position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        symbol=symbol,
                        gates=gates,
                        report_builder=build_report,
                    )

                    message_queue.put({"type": "result", "data": backtest_result})
                    message_queue.put(
                        {
                            "type": "complete",
                            "message": "LLM Strategy backtest completed!",
                        }
                    )

                except GateCancelled as e:
                    message_queue.put({"type": "cancelled", "message": str(e)})
                except Exception as e:
                    if gates is not None:
                        gates.fail(e)
                    message_queue.put(
                        {"type": "error", "message": f"Error during backtest: {str(e)}"}
                    )
                finally:
                    message_queue.put(None)  # End signal

            # Send start signal
            yield f"data: {safe_json_dumps({'type': 'start', 'message': 'Start LLM strategy backtest...'})}\n\n"

            # Start backtest in background thread
            backtest_thread = threading.Thread(target=run_backtest)
            backtest_thread.start()

            # Continuously read and send messages from the queue
            while True:
                try:
                    message = message_queue.get(timeout=1)
                    if message is None:  # End signal
                        break
                    yield f"data: {safe_json_dumps(message)}\n\n"
                except queue.Empty:
                    # Send heartbeat to maintain connection
                    yield f"data: {safe_json_dumps({'type': 'heartbeat'})}\n\n"
                    continue

            # Wait for thread completion
            backtest_thread.join()

        # Return SSE Response
        return StreamingResponse(
            generate_backtest_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unable to start streaming backtest: {str(e)}"
        )


@router.get("/runs/{run_id}")
async def get_stage_status(run_id: str):
    """Read controller-owned stage status after a run or browser disconnect."""
    try:
        return GateStore().snapshot(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Stage run not found")
    except GateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/stop")
async def stop_at_stage_boundary(run_id: str):
    """Finish the current stage and stop before beginning the next one."""
    try:
        return GateStore().request_stop(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Stage run not found")
    except GateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}/artifacts/{stage}")
async def get_stage_artifact(run_id: str, stage: Stage):
    """Read a saved handoff after verifying its content hash."""
    try:
        return GateStore().artifact(run_id, stage)
    except GateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/llm-backtest-stream/status")
async def get_stream_status():
    """
    Check streaming backtest service status
    """
    return {
        "status": "ready",
        "message": "LLM Streaming backtest service is running normally",
        "timestamp": datetime.now().isoformat(),
    }
