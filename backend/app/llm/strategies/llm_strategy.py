"""
LLM Smart Strategy
Event-driven intelligent trading strategy based on LLM
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.backtest_logger import BacktestLogger
from ...utils.indicators import calculate_bollinger_bands, calculate_macd
from ...utils.unrealized_pnl_tracker import UnrealizedPnLTracker
from ..analysis.enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
from ..analysis.trend_analyzer import EnhancedTrendAnalyzer
from ..client import OllamaError, get_llm_client
from ..decisions import (
    EXPLANATION_INSTRUCTIONS,
    DecisionError,
    format_decision_reasoning,
    validate_decision,
    execution_action,
)
from ..memory import DecisionMemory, digest, prompt_budget
from ...backtesting.gates import json_value
from .base import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
)

logger = logging.getLogger(__name__)


class LLMSmartStrategy(TradingStrategy):
    """
    LLM Smart Strategy

    Workflow:
    1. Analyze stock characteristics with historical data to determine technical indicator parameters
    2. Call LLM for decision making when key events are triggered
    3. Combine trend analysis to optimize entry and exit timing
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.llm_client = get_llm_client(temperature=0.1, max_tokens=1500, response_format="json")
        self.memory = None
        self.owner_question = "Explain your decision and what evidence would change it."
        self.trend_analyzer = EnhancedTrendAnalyzer()
        self.enhanced_analyzer = EnhancedTechnicalAnalyzer()

        # Strategy parameters
        self.confidence_threshold = config.parameters.get(
            "confidence_threshold", 0.6
        )  # Reduced to 0.6, to increase execution opportunities
        self.trend_lookback = config.parameters.get("trend_lookback", 20)
        self.event_threshold = config.parameters.get("event_threshold", 0.05)

        # Strategy type selection - Defaults to traditional
        self.strategy_type = config.parameters.get("strategy_type", "traditional")

        # Load decision principles
        self._load_strategy_prompt()
        self.max_daily_trades = config.parameters.get("max_daily_trades", 3)
        self.use_technical_filter = config.parameters.get("use_technical_filter", True)
        self.ma_short = config.parameters.get("ma_short", 10)
        self.ma_long = config.parameters.get("ma_long", 20)

        # Default technical indicator values
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.analysis_period_months = 3

        # Internal state
        self.stock_characteristics = None
        self.current_position = None
        self.decision_log = []
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.current_symbol = None  # Track current stock symbol

        # LLM Call statistics
        self.llm_call_count = 0
        self.llm_skipped_count = 0  # New: Skipped LLM call count
        self.total_events_detected = 0
        self.events_filtered_out = 0

        # Progress callback (used for streaming updates) 
        self.progress_callback = config.parameters.get("progress_callback", None)

        # Unrealized P&L tracker
        self.pnl_tracker = UnrealizedPnLTracker()
        self.current_position_id = None  # Current position ID

        # Risk management parameters
        self.max_loss_threshold = config.parameters.get(
            "max_loss_threshold", 0.10
        )  # 10%Stop loss
        self.profit_taking_threshold = config.parameters.get(
            "profit_taking_threshold", 0.15
        )  # 15%Take profit
        self.position_sizing_adjustment = config.parameters.get(
            "position_sizing_adjustment", True
        )
        self.position_size = config.parameters.get("position_size", 0.2)  # Default 20%Position allocation

        # LLM Call statistics
        self.total_llm_calls = 0
        self.events_filtered_out = 0
        self.total_events_detected = 0

        # Dynamic performance tracking
        self.initial_capital = config.parameters.get("initial_capital", 100000)
        self.current_position = None  # Current position state
        self.current_symbol = None  # Current trading symbol
        self.position_entry_price = 0.0  # Entry price
        self.position_entry_date = None  # Entry date
        self.shares = 0  # Shares held
        self.cash = self.initial_capital  # Cash balance
        self.total_trades = 0
        self.winning_trades = 0
        self.current_portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital  # Track peak value
        self.max_drawdown = 0.0
        self.total_realized_pnl = 0.0  # Cumulative realized P&L
        self.trade_returns = []  # Record returns for each trade (Percentage)

        # Risk control
        self._last_trend_analysis = None  # Store latest trend analysis for risk checks

        # Backtest logger initialization
        self.backtest_logger = None
        if config.parameters.get("enable_logging", True):
            log_path = config.parameters.get(
                "log_path", "backend/data/backtest_logs.db"
            )
            session_id = config.parameters.get("session_id", None)
            self.backtest_logger = BacktestLogger(log_path, session_id)
            logger.info(f"✅ Backtest logger enabled: {log_path}")

    def _load_strategy_prompt(self) -> None:
        """Load strategy decision principles"""
        try:
            # Determine current file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(current_dir, "prompt")

            # using traditional Strategy file
            file_path = os.path.join(prompt_dir, "traditional_strategy.md")

            # Read strategy file
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.strategy_prompt = f.read()
                logger.info(f"✅ Successfully loaded traditional Strategy: {file_path}")
            else:
                logger.warning(f"⚠️ Strategy file does not exist: {file_path}, Use default strategy")
                self.strategy_prompt = self._get_default_strategy_prompt()

        except Exception as e:
            logger.error(f"❌ Failed to load strategy file: {e}, Use default strategy")
            self.strategy_prompt = self._get_default_strategy_prompt()

    def _get_default_strategy_prompt(self) -> str:
        """Get default strategy prompt"""
        return (
            "Return JSON with action BUY/SELL/HOLD, confidence 0..1, risk_level "
            "low/medium/high, English reasoning grounded in supplied observations, "
            "and expected_outcome stating what evidence would change the decision. "
            "Avoid downtrend entry and require multiple technical confirmations."
        )

    def _send_progress(
        self,
        day: int,
        total_days: int,
        event_type: str,
        message: str,
        extra_data: dict = None,
    ):
        """Send progress updates using helper method"""
        if self.progress_callback:
            try:
                if extra_data is not None:
                    self.progress_callback(
                        day, total_days, event_type, message, extra_data
                    )
                else:
                    # Backward compatibility: If callback does not support extra_data parameters, ignore it
                    import inspect

                    sig = inspect.signature(self.progress_callback)
                    if len(sig.parameters) >= 5:
                        self.progress_callback(
                            day, total_days, event_type, message, None
                        )
                    else:
                        self.progress_callback(day, total_days, event_type, message)
            except TypeError:
                # Backward compatibility: If callback does not support 5 parameters, using 4 parameters
                self.progress_callback(day, total_days, event_type, message)

    def set_symbol(self, symbol: str):
        """Set the stock symbol currently being analyzed"""
        self.current_symbol = symbol

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate LLM intelligent trading signals

        Args:
            data: Containing OHLCV data in a DataFrame

        Returns:
            List of trading signals
        """
        signals = []

        # Validate input data
        if data is None:
            print("❌ Error: Input data is empty (None)")
            return signals

        if len(data) < 30:  # Lower data requirement
            print(f"⚠️ Insufficient data: {len(data)} < 30, Skip signal generation")
            return signals

        # Initialize P&L tracker (if not already initialized) 
        if not hasattr(self, "pnl_tracker") or self.pnl_tracker is None:
            try:
                from ...utils.unrealized_pnl_tracker import UnrealizedPnLTracker

                self.pnl_tracker = UnrealizedPnLTracker()
                print(f"📊 P&L Tracker initialized")
            except ImportError as e:
                print(f"⚠️ Unable to import P&L tracker: {e}")
                self.pnl_tracker = None

        # First step: Analyze stock characteristics (using early-period data) 
        self.stock_characteristics = self._analyze_stock_characteristics(data)

        # Dynamically adjust technical parameters based on stock characteristics
        self._adjust_technical_parameters()

        # Calculate technical indicators
        data = self._calculate_all_indicators(data)

        # Analyze trends - Add strict data length checks
        print(f"🔍 Prepare trend analysis data...")

        # Ensure sufficient data for analysis
        print(f"🔍 Data check: Total observations = {len(data)}")
        if len(data) < 50:
            print(f"⚠️ Insufficient data for trend analysis ({len(data)} < 50), Use simplified analysis")
            # Create simplified trend analysis results
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = "sideways"
            trend_analysis.complexity_score = 0.5
            trend_analysis.confidence = 0.3
        else:
            print(f"✅ Sufficient data ({len(data)} >= 50), Start full trend analysis")
            # Convert DataFrame to the required format, and validate the data
            market_data_list = []
            valid_rows = 0

            for idx, row in data.iterrows():
                # Check data completeness
                close_price = row["close"]
                if pd.isna(close_price) or close_price <= 0:
                    print(f"⚠️ Skip invalid data row: {idx}, close_price={close_price}")
                    continue

                market_data_list.append(
                    {
                        "date": idx.strftime("%Y-%m-%d")
                        if hasattr(idx, "strftime")
                        else str(idx),
                        "close": float(close_price),
                        "open": float(row["open"])
                        if "open" in row and not pd.isna(row["open"])
                        else float(close_price),
                        "high": float(row["high"])
                        if "high" in row and not pd.isna(row["high"])
                        else float(close_price),
                        "low": float(row["low"])
                        if "low" in row and not pd.isna(row["low"])
                        else float(close_price),
                        "volume": float(row["volume"])
                        if "volume" in row and not pd.isna(row["volume"])
                        else 0,
                    }
                )
                valid_rows += 1

            print(f"📊 Valid rows: {valid_rows}/{len(data)}")

            # Recheck data quantity after cleaning
            if len(market_data_list) < 30:
                print(
                    f"⚠️ Insufficient data after cleaning ({len(market_data_list)} < 30), Use simplified analysis"
                )
                from types import SimpleNamespace

                trend_analysis = SimpleNamespace()
                trend_analysis.dominant_trend = "sideways"
                trend_analysis.complexity_score = 0.5
                trend_analysis.confidence = 0.3
            else:
                print(
                    f"✅ Sufficient data after cleaning ({len(market_data_list)} >= 30), Call trend analyzer..."
                )
                print(f"🔍 Start Enhanced Trend analysis...")
                try:
                    symbol = self.current_symbol or "UNKNOWN"
                    print(f"📈 Analyze stock: {symbol}")

                    # Get current date from market data
                    current_date = None
                    if market_data_list:
                        last_data = market_data_list[-1]
                        current_date = (
                            last_data.get("date")
                            if isinstance(last_data, dict)
                            else None
                        )
                        print(f"📅 Current date: {current_date}")

                    # Ensure current_date is a string
                    current_date_str = None
                    if current_date:
                        if hasattr(current_date, "strftime"):
                            current_date_str = current_date.strftime("%Y-%m-%d")
                        elif isinstance(current_date, str):
                            current_date_str = current_date
                        else:
                            current_date_str = str(current_date)

                    enhanced_result = self.trend_analyzer.analyze_with_llm_optimization(
                        symbol, current_date_str
                    )

                    # Extract traditional trend analysis for compatibility
                    trend_analysis = enhanced_result.original_result

                    # Store enhanced results for later use in prompts
                    self.current_enhanced_analysis = enhanced_result

                    print(f"✅ Enhanced Trend analysis completed: {enhanced_result.market_phase}")
                    print(f"🎯 Reversal probability: {enhanced_result.reversal_probability:.2f}")
                    print(f"📊 Trend consistency: {enhanced_result.trend_consistency:.2f}")
                    print(f"📈 Momentum status: {enhanced_result.momentum_status}")
                    print(f"🔍 Dominant trend: {trend_analysis.dominant_trend}")

                except Exception as e:
                    print(f"❌ Enhanced Trend analysis failed: {e}")
                    import traceback

                    print(f"🔍 Error details: {traceback.format_exc()}")
                    # Create fallback analysis results
                    from types import SimpleNamespace

                    trend_analysis = SimpleNamespace()
                    trend_analysis.dominant_trend = "sideways"
                    trend_analysis.complexity_score = 0.5
                    trend_analysis.confidence = 0.2
                    self.current_enhanced_analysis = None

        print(f"🔄 Start event-driven signal generation (Data length: {len(data)})...")
        # Event-driven signal generation
        self._total_days = len(data)  # Set total days for other methods
        self._last_performance_update_day = -1  # Track the day of the last performance update, Avoid duplicates
        self._last_trend_update_day = -1  # Track the day of the last trend update

        self.analysis_rows_processed = min(30, len(data))
        for i in range(30, len(data)):  # from 30 days onward, instead of 100 days
            self.analysis_rows_processed = i + 1
            self._current_day_index = i  # Set current index for other methods

            # Periodically reanalyze the overall trend (Every 30 days or after important changes) 
            if i % 30 == 0 and i != self._last_trend_update_day:
                print(f"🔄 Day{i} days: Reanalyze overall trend...")
                try:
                    current_data_for_trend = data.iloc[: i + 1].copy()
                    market_data_list = []

                    # Rebuild market data
                    for idx, row in current_data_for_trend.iterrows():
                        market_data_list.append(
                            {
                                "date": idx,
                                "open": row.get("open", row.get("Open", 0)),
                                "high": row.get("high", row.get("High", 0)),
                                "low": row.get("low", row.get("Low", 0)),
                                "close": row.get("close", row.get("Close", 0)),
                                "volume": row.get("volume", row.get("Volume", 0)),
                            }
                        )

                    if len(market_data_list) >= 30:
                        symbol = self.current_symbol or "UNKNOWN"
                        current_date = market_data_list[-1].get("date")

                        # Ensure current_date is a string
                        if hasattr(current_date, "strftime"):
                            current_date_str = current_date.strftime("%Y-%m-%d")
                        elif isinstance(current_date, str):
                            current_date_str = current_date
                        else:
                            current_date_str = str(current_date)

                        enhanced_result = (
                            self.trend_analyzer.analyze_with_llm_optimization(
                                symbol, current_date_str
                            )
                        )
                        trend_analysis = enhanced_result.original_result
                        self.current_enhanced_analysis = enhanced_result

                        print(f"📊 Update overall trend: {enhanced_result.market_phase}")
                        print(
                            f"🎯 Reversal probability: {enhanced_result.reversal_probability:.2f}"
                        )
                        print(f"📈 Momentum status: {enhanced_result.momentum_status}")

                        self._last_trend_update_day = i

                except Exception as e:
                    print(f"⚠️ Overall trend update failed: {e}")

            if i % 50 == 0:  # Every 50 days between progress reports
                progress_percentage = (i / len(data) * 100) if len(data) > 0 else 0
                progress_msg = (
                    f"📊 Processing progress: {i}/{len(data)} ({progress_percentage:.1f}%)"
                )
                print(progress_msg)

                # If a progress callback is available, Send progress update
                if self.progress_callback:
                    self.progress_callback(
                        i, len(data), "processing", progress_msg, None
                    )

            # Every 10 days between performance updates (Including P&L State), Avoid duplicate updates after trades
            if (
                self.progress_callback
                and i % 10 == 0
                and i != self._last_performance_update_day
            ):
                current_row = data.iloc[i]
                current_price = current_row.get("close", current_row.get("Close", 0))
                if current_price > 0:
                    self._send_performance_update(i, len(data), current_price)
                    self._last_performance_update_day = i

            current_date = data.index[i]
            historical_data = data.iloc[: i + 1]

            # Safely get timestamp - Handle possible integer indexes
            try:
                if hasattr(current_date, "date"):
                    # Is a datetime object
                    timestamp = current_date
                    current_date_obj = current_date.date()
                else:
                    # Is an integer index, Create a default timestamp
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                    current_date_obj = i

                if self.last_trade_date != current_date_obj:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date_obj
            except Exception:
                # If date handling fails, Use default value
                timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                current_date_obj = i
                if self.last_trade_date != current_date_obj:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date_obj

            if self.daily_trade_count >= self.max_daily_trades:
                continue

            # Detect trigger events
            events = self._detect_trigger_events(historical_data, i)
            self.total_events_detected += len(events)

            if events:
                current_price = historical_data.iloc[i]["close"]
                current_date = historical_data.index[i].strftime("%Y-%m-%d")

                # Show current P&L State
                if self.current_position:
                    position_metrics = self._calculate_position_metrics(
                        current_price, historical_data.index[i]
                    )
                    unrealized_pnl = position_metrics.get("unrealized_pnl", 0)
                    unrealized_pnl_pct = position_metrics.get("unrealized_pnl_pct", 0)
                    holding_days = position_metrics.get("holding_days", 0)

                    print(f"🎯 Day{i} events detected on day: {[e['event_type'] for e in events]}")
                    print(
                        f"💰 Position held: {self.shares} shares@${self.position_entry_price:.2f}, Current price${current_price:.2f}"
                    )
                    print(
                        f"📊 Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%), Holding{holding_days} days"
                    )
                else:
                    print(f"🎯 Day{i} events detected on day: {[e['event_type'] for e in events]}")
                    print(
                        f"💵 No position held: Cash${self.cash:,.2f}, Current price${current_price:.2f}"
                    )

                # Optimization 1: No position + Sideways market + No strong signal only then skip LLM
                if (
                    not self.current_position
                    and trend_analysis.dominant_trend == "sideways"
                ):
                    # Check for strong technical signals
                    strong_signals = [
                        e
                        for e in events
                        if e["event_type"]
                        in [
                            "BB_LOWER_TOUCH",
                            "BB_UPPER_TOUCH",
                            "MACD_GOLDEN_CROSS",
                            "MACD_DEATH_CROSS",
                        ]
                    ]
                    if not strong_signals:
                        print(
                            f"⏭️ No position+Sideways market+No strong signal, Skip LLM Decision (Trend: {trend_analysis.dominant_trend})"
                        )
                        self.events_filtered_out += len(events)
                        continue
                    else:
                        print(
                            f"✅ Sideways market with strong signal {[s['event_type'] for s in strong_signals]}, Continue LLM Decision"
                        )

                # Optimization 2: Filter relevant events by position state
                relevant_events = self._filter_relevant_events(
                    events, self.current_position
                )
                self.events_filtered_out += len(events) - len(relevant_events)

                if not relevant_events:
                    print(
                        f"⏭️ No relevant events (Position held: {'Position held' if self.current_position else 'No position'}), Skip LLM Decision"
                    )
                    continue

                # Optimization 3: Additional check - When holding in an uptrend without a long bearish K candle or other strong signals, skip as well
                if self.current_position and trend_analysis.dominant_trend == "uptrend":
                    # Check for a long bearish K candle or other strong exit signals
                    has_large_drop = any(
                        event.get("event_type") == "LARGE_DROP"
                        for event in relevant_events
                    )
                    has_strong_exit_signal = any(
                        event.get("severity") == "high" for event in relevant_events
                    )

                    if not (has_large_drop or has_strong_exit_signal):
                        print(f"⏭️ Holding+Uptrend+No long bearish K candle or strong signal, Skip LLM Decision")
                        continue

                print(
                    f"📋 Relevant events: {[e['event_type'] for e in relevant_events]} (Original events: {len(events)}, After filtering: {len(relevant_events)})"
                )
                print(
                    f"📈 Current trend: {trend_analysis.dominant_trend}, Holding: {'Position held' if self.current_position else 'No position'}"
                )

                # Decide whether to call based on trend and position state LLM
                skip_llm = False
                skip_reason = ""

                has_position = self.current_position is not None

                if has_position:
                    # Position held: Risk management first
                    if trend_analysis.dominant_trend == "downtrend":
                        # Holding + Downtrend: Requires LLM stop-loss analysis/Exit strategy
                        skip_llm = False
                        print(f"⚠️  Holding during a downtrend, calls LLM Analyze stop-loss strategy")
                    elif trend_analysis.dominant_trend == "sideways":
                        # Holding + Consolidation: Requires LLM Find a suitable exit point
                        skip_llm = False
                        print(f"📊 Holding during consolidation, calls LLM Find the best exit point")
                    elif trend_analysis.dominant_trend == "uptrend":
                        # Holding + Uptrend: Only a long bearish K candle requires LLM assessment
                        has_large_drop = any(
                            event["event_type"] == "LARGE_DROP"
                            for event in relevant_events
                        )
                        if has_large_drop:
                            skip_llm = False
                            print(
                                f"⚠️  Holding+Uptrend+Long bearish K candle, calls LLM Analyze profit-taking opportunities"
                            )
                        else:
                            skip_llm = True
                            skip_reason = "Holding+Uptrend+No long bearish K candle, Continue holding"
                else:
                    # No position held: Entry timing selection
                    if trend_analysis.dominant_trend == "downtrend":
                        # No position + Downtrend: Call only for strong reversal signals LLM
                        has_reversal_signal = any(
                            event.get("event_type")
                            in ["REVERSAL_PATTERN", "SUPPORT_BOUNCE"]
                            for event in relevant_events
                        )
                        if has_reversal_signal:
                            skip_llm = False
                            print(f"🔄 Reversal signal found in downtrend, calls LLM Analyze dip-buying opportunity")
                        else:
                            skip_llm = True
                            skip_reason = "No position+Downtrend+No reversal signal, Avoid countertrend trading"
                    elif trend_analysis.dominant_trend == "sideways":
                        # No position + Consolidation: Relax conditions, Increase entry opportunities
                        has_breakout_signal = any(
                            event.get("event_type")
                            in [
                                "BREAKOUT",
                                "VOLUME_SPIKE",
                                "MOMENTUM_SHIFT",
                                "TREND_TURN_BULLISH",
                                "TREND_TURN_BEARISH",
                            ]
                            for event in relevant_events
                        )
                        has_strong_reversal = any(
                            event.get("event_type")
                            in ["BB_LOWER_TOUCH", "BB_UPPER_TOUCH"]
                            and event.get("severity") in ["high", "very_high"]
                            for event in relevant_events
                        )
                        has_macd_signal = any(
                            event.get("event_type")
                            in ["MACD_GOLDEN_CROSS", "MACD_DEATH_CROSS"]
                            for event in relevant_events
                        )
                        has_ma_signal = any(
                            event.get("event_type")
                            in ["MA_GOLDEN_CROSS", "MA_DEATH_CROSS"]
                            for event in relevant_events
                        )
                        has_multiple_signals = (
                            len(relevant_events) >= 2
                        )  # Multiple simultaneous technical signals

                        # Relax conditions: Any technical signal merits LLM analysis
                        if (
                            has_breakout_signal
                            or has_strong_reversal
                            or has_macd_signal
                            or has_ma_signal
                            or has_multiple_signals
                        ):
                            skip_llm = False
                            signal_types = [
                                event["event_type"] for event in relevant_events
                            ]
                            print(
                                f"✅ Technical signal detected in sideways market {signal_types}, calls LLM Analyze opportunity"
                            )
                        else:
                            skip_llm = True
                            skip_reason = "No position+Consolidating trend, Wait for a clear breakout signal"
                    elif trend_analysis.dominant_trend == "uptrend":
                        # No position + Uptrend: Call normally LLM Analyze entry opportunity
                        skip_llm = False
                        print(f"🚀 No position during uptrend, calls LLM Analyze entry opportunity")

                if skip_llm:
                    # Do not call LLM, but record events and reasons
                    self.llm_skipped_count += 1  # Increment skip counter
                    event_summary = ", ".join(
                        [e["event_type"] for e in relevant_events]
                    )
                    skip_msg = f"⏭️ {timestamp.strftime('%Y-%m-%d')} {skip_reason} (Events detected: {event_summary})"
                    print(skip_msg)

                    # Send skip progress message
                    if self.progress_callback:
                        self.progress_callback(
                            i, len(data), "llm_skipped", skip_msg, None
                        )
                    continue

                # Reanalyze trend at the current point in time - using Enhanced analysis
                print(f"🔍 Reanalyze {timestamp.strftime('%Y-%m-%d')} trend...")

                # Prefer Enhanced analysis, fallback to original analysis
                current_enhanced_analysis = None
                try:
                    # using Enhanced Trend Analyzer Analyze at the current point in time
                    symbol = self.current_symbol or "UNKNOWN"
                    current_date_str = timestamp.strftime("%Y-%m-%d")
                    current_enhanced_analysis = (
                        self.trend_analyzer.analyze_with_llm_optimization(
                            symbol, current_date_str
                        )
                    )
                    self.current_enhanced_analysis = current_enhanced_analysis
                    print(
                        f"✅ Enhanced Trend analysis: {current_enhanced_analysis.market_phase}"
                    )
                    print(f"📊 Momentum status: {current_enhanced_analysis.momentum_status}")
                    print(
                        f"🎯 Reversal probability: {current_enhanced_analysis.reversal_probability:.3f}"
                    )

                    # Create compatible trend_analysis object for other code
                    current_trend_analysis = current_enhanced_analysis.original_result

                except Exception as e:
                    print(f"⚠️ Enhanced Analysis failed, Fall back to simplified analysis: {e}")
                    current_trend_analysis = self._analyze_current_trend(
                        historical_data, timestamp
                    )
                    current_enhanced_analysis = None

                self._last_trend_analysis = current_trend_analysis  # Store for risk checks

                if current_enhanced_analysis:
                    print(
                        f"📊 Enhanced Trend analysis: {current_enhanced_analysis.market_phase}"
                    )
                else:
                    print(f"📊 Simplified trend analysis: {current_trend_analysis.dominant_trend}")

                # Call LLM Make decision
                self.llm_call_count += 1  # Increment counter

                # Send LLM decision-start progress message
                if self.progress_callback:
                    llm_start_msg = (
                        f"🤖 {timestamp.strftime('%Y-%m-%d')} Start LLM analysis..."
                    )
                    self.progress_callback(
                        i, len(data), "llm_decision", llm_start_msg, None
                    )

                llm_decision = self._make_llm_decision(
                    historical_data,
                    timestamp,  # Use processed timestamp
                    relevant_events,  # Use filtered events
                    current_trend_analysis,  # Pass compatible analysis result, but prompt will use enhanced
                )

                # Send LLM Decision results
                if llm_decision:
                    action = llm_decision.get("action", "HOLD")
                    confidence = llm_decision.get("confidence", 0)
                    reason = llm_decision.get("reasoning", "No explanation")
                    decision_msg = f"🤖 {timestamp.strftime('%Y-%m-%d')} LLM Decision: {action} (Confidence: {confidence:.2f}) - {reason}"
                    print(decision_msg)

                    if self.progress_callback:
                        self.progress_callback(
                            i, len(data), "llm_decision", decision_msg, None
                        )
                else:
                    print(f"🤖 {timestamp.strftime('%Y-%m-%d')} LLM Decision: No clear recommendation")

                # Record logs - Daily analysis data
                if self.backtest_logger:
                    self._log_daily_analysis(
                        timestamp=timestamp,
                        historical_data=historical_data,
                        i=i,
                        events=events,
                        relevant_events=relevant_events,
                        trend_analysis=current_trend_analysis,
                        llm_decision=llm_decision,
                        comprehensive_context=getattr(
                            self, "current_comprehensive_context", None
                        ),
                    )

                if execution_action(llm_decision) in ["BUY", "SELL"]:
                    # Check confidence threshold
                    confidence = llm_decision.get("confidence", 0)
                    if confidence >= self.confidence_threshold:
                        # Get current price
                        current_price = historical_data.iloc[-1]["close"]

                        # Use original decision
                        enhanced_decision = llm_decision.copy()

                        signal = self._create_signal_from_decision(
                            enhanced_decision,
                            timestamp,  # Use processed timestamp
                            current_price,
                        )
                        if signal:
                            signals.append(signal)
                            self.daily_trade_count += 1
                            signal_msg = f"✅ Generate trading signals: {signal.signal_type} (Confidence: {confidence:.2f} >= Threshold: {self.confidence_threshold:.2f})"
                            print(signal_msg)

                            # Log trading signal
                            if self.backtest_logger:
                                self._log_trading_signal(
                                    timestamp, signal, llm_decision
                                )

                            # Calculate current P&L state for frontend display
                            pnl_data = {}
                            if hasattr(self, "pnl_tracker") and self.pnl_tracker:
                                try:
                                    current_row = data.iloc[i]
                                    current_price = current_row.get(
                                        "close", current_row.get("Close", 0)
                                    )
                                    position_metrics = self._calculate_position_metrics(
                                        current_price, current_date
                                    )
                                    if position_metrics and position_metrics.get(
                                        "has_position"
                                    ):
                                        pnl_data = {
                                            "unrealized_pnl": position_metrics.get(
                                                "unrealized_pnl", 0
                                            ),
                                            "unrealized_pnl_pct": position_metrics.get(
                                                "unrealized_pnl_pct", 0
                                            ),
                                            "holding_days": position_metrics.get(
                                                "holding_days", 0
                                            ),
                                            "shares": position_metrics.get("shares", 0),
                                            "risk_level": position_metrics.get(
                                                "risk_level", "normal"
                                            ),
                                            "cash_remaining": self.cash,
                                            "total_value": self.cash
                                            + (
                                                position_metrics.get("shares", 0)
                                                * current_price
                                            ),
                                        }
                                    else:
                                        pnl_data = {
                                            "unrealized_pnl": 0,
                                            "unrealized_pnl_pct": 0,
                                            "holding_days": 0,
                                            "shares": 0,
                                            "risk_level": "normal",
                                            "cash_remaining": self.cash,
                                            "total_value": self.cash,
                                        }
                                except Exception as e:
                                    print(f"⚠️ P&L Calculation failed: {e}")

                            # Send trading-signal generation progress, Containing P&L Information
                            if self.progress_callback:
                                extra_data = (
                                    {"pnl_status": pnl_data} if pnl_data else None
                                )
                                self.progress_callback(
                                    i,
                                    len(data),
                                    "signal_generated",
                                    signal_msg,
                                    extra_data,
                                )

                                # Send performance update immediately after signal generation
                                current_row = data.iloc[i]
                                current_price = current_row.get(
                                    "close", current_row.get("Close", 0)
                                )
                                if current_price > 0:
                                    self._send_performance_update(
                                        i, len(data), current_price
                                    )
                    else:
                        print(
                            f"❌ Insufficient confidence: {llm_decision.get('confidence', 0):.2f} < {self.confidence_threshold}"
                        )

        print(f"🎉 Signal generation completed! Total generated {len(signals)} signals")

        # Print optimization statistics
        print(f"")
        print(f"📊 LLM Call optimization statistics:")
        print(f"   📈 Total trading days: {len(data)} days")
        print(f"   🎯 Total detected events: {self.total_events_detected} items")
        print(f"   🗑️ Filtered events: {self.events_filtered_out} items")
        print(f"   🤖 LLM Actual calls: {self.llm_call_count} times")
        print(f"   ⏭️  LLM Skipped calls: {self.llm_skipped_count} times (Decline/Consolidating trend)")

        # Calculate efficiency safely, Avoid division by zero
        data_length = len(data) if len(data) > 0 else 1
        print(f"   ⚡ Actual call efficiency: {self.llm_call_count / data_length:.3f} times/days")

        total_potential_calls = self.llm_call_count + self.llm_skipped_count
        if total_potential_calls > 0:
            print(
                f"   🎯 Trend filter rate: {self.llm_skipped_count / total_potential_calls:.1%}"
            )
        if self.total_events_detected > 0:
            print(
                f"   🎯 Event processing rate: {(self.total_events_detected - self.events_filtered_out) / self.total_events_detected:.1%}"
            )
        print(
            f"   💰 Cost savings: {(1 - self.llm_call_count / data_length) * 100:.1f}% (Compared with daily calls)"
        )
        print(
            f"   💡 Intelligent savings: {(1 - self.llm_call_count / (self.llm_call_count + self.llm_skipped_count)) * 100:.1f}% (Compared with calls for all events)"
            if total_potential_calls > 0
            else ""
        )

        return signals

    def _analyze_stock_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze stock characteristics in depth (Determine stock behavior from early-period data) 

        Args:
            data: Stock data

        Returns:
            Stock characteristics analysis results
        """
        # Use the first 3-6 months of data for analysis, but require at least 60 days
        analysis_days = max(60, self.analysis_period_months * 30)
        analysis_data = data.iloc[: min(analysis_days, len(data) // 2)]

        if len(analysis_data) < 30:
            analysis_data = data.iloc[:30] if len(data) >= 30 else data

        print(f"📈 Analyze stock characteristics (using {len(analysis_data)} days of historical data) ...")

        # Calculate basic statistical characteristics
        returns = analysis_data["close"].pct_change().dropna()
        prices = analysis_data["close"]

        # Ensure sufficient data for calculations
        if len(returns) < 2:
            logger.warning(
                f"Insufficient data for analysis: only {len(returns)} return values"
            )
            return None

        # 1. Volatility analysis (Multidimensional)  - Correct pandas syntax
        daily_volatility = returns.std() if len(returns) > 1 else 0.0
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Calculate volatility of volatility, Ensure sufficient rolling-window data
        rolling_volatility = returns.rolling(10, min_periods=5).std()
        volatility_of_volatility = (
            rolling_volatility.std() if len(rolling_volatility.dropna()) > 1 else 0.0
        )

        # 2. Analyze trend characteristics
        trend_consistency = self._calculate_trend_consistency(analysis_data)
        trend_strength = self._calculate_trend_strength(analysis_data)

        # 3. Price action analysis - Calculate safely to avoid division by zero and NaN
        price_mean = prices.mean()
        if price_mean > 0:
            price_range_ratio = (prices.max() - prices.min()) / price_mean
        else:
            price_range_ratio = 0.0

        avg_daily_return = returns.mean() if len(returns) > 0 else 0.0
        skewness = (
            returns.skew() if len(returns) > 2 else 0.0
        )  # Skewness: Positive values indicate right skew (more upward moves) 
        kurtosis = (
            returns.kurtosis() if len(returns) > 3 else 0.0
        )  # Kurtosis: Measures frequency of extreme values

        # 4. Reversal characteristics
        reversal_frequency = self._calculate_reversal_frequency(analysis_data)
        consecutive_days = self._calculate_consecutive_move_tendency(returns)

        # 5. Volume characteristics - Calculate safely to avoid NaN
        volume_volatility = 0.0
        volume_price_correlation = 0.0
        if "volume" in analysis_data.columns and len(analysis_data["volume"]) > 1:
            volume_changes = analysis_data["volume"].pct_change().dropna()
            if len(volume_changes) > 1:
                volume_volatility = volume_changes.std()
                # Calculate correlation safely, Ensure indexes align
                common_index = returns.index.intersection(volume_changes.index)
                if len(common_index) > 1:
                    aligned_returns = returns.reindex(common_index)
                    aligned_volume = volume_changes.reindex(common_index)
                    volume_price_correlation = aligned_returns.corr(aligned_volume)

        # 6. Test technical indicator responsiveness
        macd_effectiveness = self._test_macd_effectiveness(analysis_data)
        ma_crossover_effectiveness = self._test_ma_crossover_effectiveness(
            analysis_data
        )
        bb_effectiveness = self._test_bollinger_bands_effectiveness(analysis_data)

        # 7. Support and resistance analysis
        support_resistance_strength = self._analyze_support_resistance(analysis_data)
        breakout_tendency = self._analyze_breakout_tendency(analysis_data)

        # 8. Stock behavior classification
        stock_personality = self._classify_stock_personality(
            annualized_volatility,
            trend_consistency,
            reversal_frequency,
            macd_effectiveness,
        )

        characteristics = {
            # Volatility indicators
            "volatility": annualized_volatility,
            "daily_volatility": daily_volatility,
            "volatility_of_volatility": volatility_of_volatility,
            # Return characteristics
            "avg_daily_return": avg_daily_return,
            "annualized_return": avg_daily_return * 252,
            "sharpe_ratio": avg_daily_return / daily_volatility
            if daily_volatility > 0
            else 0,
            "skewness": skewness,
            "kurtosis": kurtosis,
            # Trend characteristics
            "trend_consistency": trend_consistency,
            "trend_strength": trend_strength,
            "reversal_frequency": reversal_frequency,
            "consecutive_move_tendency": consecutive_days,
            # Price action
            "price_range_ratio": price_range_ratio,
            "breakout_tendency": breakout_tendency,
            # Volume characteristics
            "volume_volatility": volume_volatility,
            "volume_price_correlation": volume_price_correlation,
            # Technical indicator responsiveness
            "macd_effectiveness": macd_effectiveness,
            "ma_crossover_effectiveness": ma_crossover_effectiveness,
            "bollinger_effectiveness": bb_effectiveness,
            # Support and resistance
            "support_resistance_strength": support_resistance_strength,
            # Overall classification
            "stock_personality": stock_personality,
        }

        # Print analysis results
        print(f"📊 Stock characteristics analysis completed:")
        print(f"   Annualized volatility: {annualized_volatility:.1%}")
        print(f"   Annualized return: {characteristics['annualized_return']:.1%}")
        print(f"   Sharpe ratio: {characteristics['sharpe_ratio']:.2f}")
        print(f"   Trend consistency: {trend_consistency:.2f}")
        print(f"   Reversal frequency: {reversal_frequency:.2f}")
        print(f"   Stock behavior classification: {stock_personality}")
        print(f"   MACD Effectiveness: {macd_effectiveness:.2f}")

        return characteristics

    def _adjust_technical_parameters(self):
        """Adjust technical parameters based on stock characteristics"""
        if not self.stock_characteristics:
            return

        print(f"📊 Stock characteristics analysis results:")
        print(f"   Volatility: {self.stock_characteristics['volatility']:.3f}")
        print(f"   Trend consistency: {self.stock_characteristics['trend_consistency']:.3f}")
        print(f"   Reversal frequency: {self.stock_characteristics['reversal_frequency']:.3f}")
        print(
            f"   MACD Effectiveness: {self.stock_characteristics.get('macd_effectiveness', 0.5):.3f}"
        )

        # Save original parameters as a baseline
        original_ma_short = self.ma_short
        original_ma_long = self.ma_long
        original_macd_fast = self.macd_fast
        original_macd_slow = self.macd_slow

        # 1. Adjust based on trend consistency MACD parameters
        trend_consistency = self.stock_characteristics["trend_consistency"]
        print(f"🔍 Adjust MACD parameters - Trend consistency: {trend_consistency:.3f}")
        if trend_consistency > 0.8:  # Extremely strong trend
            self.macd_fast = 6  # Capture trends quickly
            self.macd_slow = 18
            print(f"   Extremely strong trend -> MACD Set to 6/18")
        elif trend_consistency > 0.6:  # Strong trend
            self.macd_fast = 8
            self.macd_slow = 21
            print(f"   Strong trend -> MACD Set to 8/21")
        elif trend_consistency > 0.4:  # Moderate trend
            self.macd_fast = 12  # Standard settings
            self.macd_slow = 26
            print(f"   Moderate trend -> MACD Set to 12/26")
        elif trend_consistency > 0.2:  # Weak trend, Mostly sideways
            self.macd_fast = 15
            self.macd_slow = 35
            print(f"   Weak trend -> MACD Set to 15/35")
        else:  # Strongly range-bound
            self.macd_fast = 20  # Long period, Reduce false signals
            self.macd_slow = 45
            print(f"   Strongly range-bound -> MACD Set to 20/45")

        # 3. Adjust moving-average parameters based on reversal frequency
        reversal_freq = self.stock_characteristics["reversal_frequency"]
        print(f"🔍 Adjust moving-average parameters - Reversal frequency: {reversal_freq:.3f}")
        if reversal_freq > 0.15:  # High reversal frequency - Range-bound stock
            self.ma_short = max(5, self.ma_short - 2)  # Shorten period
            self.ma_long = max(15, self.ma_long - 5)
            print(f"   High reversal frequency -> Shorten moving-average period")
        elif reversal_freq < 0.05:  # Low reversal frequency - Trending stock
            self.ma_short = min(20, self.ma_short + 3)  # Lengthen period
            self.ma_long = min(50, self.ma_long + 10)
            print(f"   Low reversal frequency -> Lengthen moving-average period")

        # 4. Fine-tune based on technical indicator effectiveness
        macd_effectiveness = self.stock_characteristics.get("macd_effectiveness", 0.5)

        # If MACD Poor effectiveness, Use more conservative parameters
        if macd_effectiveness < 0.4:
            self.macd_fast = min(20, int(self.macd_fast * 1.2))
            self.macd_slow = min(50, int(self.macd_slow * 1.1))
            print(f"   MACD Poor effectiveness -> Conservative parameters {self.macd_fast}/{self.macd_slow}")

        # 5. Price range adjustment - Account for stock price fluctuation range
        print(f"\n🔧 Adaptive technical parameter adjustment:")
        print(f"   MACD Fast line: {original_macd_fast} → {self.macd_fast}")
        print(f"   MACD Slow line: {original_macd_slow} → {self.macd_slow}")
        print(f"   Short-term moving average: {original_ma_short} → {self.ma_short}")
        print(f"   Long-term moving average: {original_ma_long} → {self.ma_long}")

        # Ensure reasonable parameters
        self.macd_fast = max(3, min(20, self.macd_fast))
        self.macd_slow = max(10, min(50, self.macd_slow))
        self.ma_short = max(3, min(20, self.ma_short))
        self.ma_long = max(10, min(50, self.ma_long))

        # Ensure fast line < Slow line
        if self.macd_fast >= self.macd_slow:
            self.macd_slow = self.macd_fast + 5
        if self.ma_short >= self.ma_long:
            self.ma_long = self.ma_short + 5

        print(f"\n✅ Final parameters (after range checks) :")
        print(f"   MACD: {self.macd_fast}/{self.macd_slow}")
        print(f"   Moving average: {self.ma_short}/{self.ma_long}")

    def _calculate_position_metrics(
        self, current_price: float, current_date: pd.Timestamp = None
    ) -> Dict[str, Any]:
        """Calculate detailed metrics for the current position"""
        if not self.current_position:
            return {
                "has_position": False,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "holding_days": 0,
                "shares": 0,
                "position_value": 0.0,
                "risk_level": "normal",
            }

        # Calculate unrealized P&L
        position_value = self.shares * current_price
        unrealized_pnl = position_value - (self.shares * self.position_entry_price)
        unrealized_pnl_pct = (
            unrealized_pnl / (self.shares * self.position_entry_price) * 100
        )

        # Calculate holding days
        if self.position_entry_date and current_date:
            if isinstance(self.position_entry_date, str):
                entry_date = datetime.strptime(self.position_entry_date, "%Y-%m-%d")
            elif hasattr(self.position_entry_date, "date"):
                entry_date = self.position_entry_date
            else:
                entry_date = pd.to_datetime(self.position_entry_date)

            if hasattr(current_date, "date"):
                current_date_obj = current_date
            else:
                current_date_obj = pd.to_datetime(current_date)

            holding_days = (current_date_obj - entry_date).days
        else:
            holding_days = 0

        # Risk level assessment
        risk_level = self._assess_risk_level(unrealized_pnl_pct, holding_days)

        return {
            "has_position": True,
            "entry_price": self.position_entry_price,
            "current_price": current_price,
            "shares": self.shares,
            "position_value": position_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "holding_days": holding_days,
            "risk_level": risk_level,
            "cost_basis": self.shares * self.position_entry_price,
        }

    def _calculate_current_performance(self, current_price: float) -> Dict[str, float]:
        """Calculate current overall performance metrics"""
        # Calculate current total value
        position_value = self.shares * current_price if self.shares > 0 else 0
        total_value = self.cash + position_value

        # Calculate total return
        total_return = (total_value - self.initial_capital) / self.initial_capital

        # Calculate win rate
        win_rate = (
            self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        )

        # Calculate cumulative trade returns (Sum of individual trade returns)
        cumulative_trade_return_rate = sum(self.trade_returns) / 100  # Convert to decimal form

        # Update peak value and drawdown
        if total_value > self.max_portfolio_value:
            self.max_portfolio_value = total_value

        # Calculate current drawdown
        current_drawdown = (
            (self.max_portfolio_value - total_value) / self.max_portfolio_value
            if self.max_portfolio_value > 0
            else 0
        )
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "total_value": total_value,
            "cash": self.cash,
            "position_value": position_value,
            "total_realized_pnl": self.total_realized_pnl,  # Cumulative realized P&L
            "cumulative_trade_return_rate": cumulative_trade_return_rate,  # Cumulative trade return
        }

    def _send_performance_update(self, day: int, total_days: int, current_price: float):
        """Send performance update message"""
        if not self.progress_callback:
            return

        performance = self._calculate_current_performance(current_price)

        # Calculate current P&L State
        pnl_status = None
        if hasattr(self, "pnl_tracker") and self.pnl_tracker and current_price > 0:
            try:
                position_metrics = self._calculate_position_metrics(current_price)
                if position_metrics["has_position"]:
                    pnl_status = {
                        "unrealized_pnl": position_metrics["unrealized_pnl"],
                        "unrealized_pnl_pct": position_metrics["unrealized_pnl_pct"],
                        "holding_days": position_metrics["holding_days"],
                        "shares": position_metrics["shares"],
                        "risk_level": position_metrics["risk_level"],
                        "cash_remaining": self.cash,
                        "total_value": self.cash + position_metrics["position_value"],
                    }
                else:
                    # Send complete information even without a position P&L State
                    pnl_status = {
                        "unrealized_pnl": 0,
                        "unrealized_pnl_pct": 0,
                        "holding_days": 0,
                        "shares": 0,
                        "risk_level": "normal",
                        "cash_remaining": self.cash,
                        "total_value": self.cash,
                    }
            except Exception as e:
                print(f"⚠️ P&L Calculation failed: {e}")

        # Build a message matching the frontend format
        message = f"Total return: {performance['total_return'] * 100:+.2f}% | Win rate: {performance['win_rate'] * 100:.1f}% | Maximum drawdown: {performance['max_drawdown'] * 100:.2f}%"

        # Also in extra_data send detailed data
        extra_data = {"performance_metrics": performance, "pnl_status": pnl_status}

        self._send_progress(day, total_days, "performance_update", message, extra_data)

    def _assess_risk_level(self, pnl_pct: float, holding_days: int) -> str:
        """Assess risk level of the current position"""
        if pnl_pct <= -self.max_loss_threshold * 100:
            return "high_loss"  # High loss risk
        elif pnl_pct <= -2:
            return "moderate_loss"  # Moderate loss
        elif pnl_pct >= self.profit_taking_threshold * 100:
            return "high_profit"  # High profit
        elif pnl_pct >= 8:
            return "moderate_profit"  # Moderate profit
        elif holding_days > 30:
            return "long_hold"  # Long-held position
        else:
            return "normal"  # Normal state

    def _generate_pnl_insights(
        self, position_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investment insights from unrealized P&L"""
        if not position_metrics["has_position"]:
            return {
                "pnl_signal": "neutral",
                "risk_warning": None,
                "suggested_action": "Consider a new position",
                "position_sizing_factor": 1.0,
            }

        pnl_pct = position_metrics["unrealized_pnl_pct"]
        risk_level = position_metrics["risk_level"]
        holding_days = position_metrics["holding_days"]

        insights = {
            "pnl_signal": "neutral",
            "risk_warning": None,
            "suggested_action": "Continue holding",
            "position_sizing_factor": 1.0,
        }

        # Provide suggestions based on P&L
        if risk_level == "high_loss":
            insights.update(
                {
                    "pnl_signal": "stop_loss",
                    "risk_warning": f"Loss has reached{pnl_pct:.1f}%, Consider a stop loss",
                    "suggested_action": "Evaluate stop loss immediately",
                    "position_sizing_factor": 0.5,
                }
            )
        elif risk_level == "moderate_loss":
            insights.update(
                {
                    "pnl_signal": "caution",
                    "risk_warning": f"Current loss{pnl_pct:.1f}%, Proceed cautiously",
                    "suggested_action": "Carefully evaluate the next strategy",
                    "position_sizing_factor": 0.7,
                }
            )
        elif risk_level == "high_profit":
            insights.update(
                {
                    "pnl_signal": "take_profit",
                    "risk_warning": None,
                    "suggested_action": f"Profit has reached{pnl_pct:.1f}%, Consider taking profit",
                    "position_sizing_factor": 0.8,
                }
            )
        elif risk_level == "moderate_profit":
            insights.update(
                {
                    "pnl_signal": "bullish",
                    "risk_warning": None,
                    "suggested_action": f"Profit{pnl_pct:.1f}%, Performing well",
                    "position_sizing_factor": 1.2,
                }
            )
        elif risk_level == "long_hold":
            insights.update(
                {
                    "pnl_signal": "review",
                    "risk_warning": f"Position held for{holding_days} days, Reassessment recommended",
                    "suggested_action": "Review whether the holding strategy needs adjustment",
                    "position_sizing_factor": 0.9,
                }
            )

        return insights

    def _update_position_state(
        self, action: str, price: float, quantity: int, date: str
    ):
        """Update position state"""
        if action == "BUY":
            if self.current_position is None:
                # Open new position
                self.current_position = "long"
                self.position_entry_price = price
                self.position_entry_date = date
                self.shares = quantity
                self.cash -= quantity * price
                print(f"📈 Open position: {quantity} shares @ ${price:.2f}")
            else:
                # Add to position (Simplified for now, Use simple average cost)
                total_cost = self.shares * self.position_entry_price + quantity * price
                self.shares += quantity
                self.position_entry_price = total_cost / self.shares
                self.cash -= quantity * price
                print(
                    f"📈 Add to position: +{quantity} shares @ ${price:.2f}, Average cost: ${self.position_entry_price:.2f}"
                )

        elif action == "SELL":
            if self.current_position is not None:
                # Calculate realized P&L
                sell_value = quantity * price
                cost_basis = quantity * self.position_entry_price
                realized_pnl = sell_value - cost_basis
                realized_pnl_pct = (realized_pnl / cost_basis) * 100

                self.cash += sell_value
                self.shares -= quantity

                if self.shares <= 0:
                    # Close entire position
                    print(
                        f"📉 Close position: {quantity} shares @ ${price:.2f}, Realized P&L: ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)"
                    )
                    self.current_position = None
                    self.position_entry_price = 0.0
                    self.position_entry_date = None
                    self.shares = 0
                else:
                    # Close partial position
                    print(
                        f"📉 Reduce position: -{quantity} shares @ ${price:.2f}, Realized P&L: ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)"
                    )

                # Note: Trade statistics are updated in _create_signal_from_decision updated there, Do not update again here

    def calculate_position_size(self, price: float) -> int:
        """Calculate suggested position size - Fixed 1000 shares"""
        return 1000

    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        data = data.copy()

        # MACD
        macd_data = calculate_macd(
            data,
            short_period=self.macd_fast,
            long_period=self.macd_slow,
            signal_period=self.macd_signal,
        )
        data["macd"] = macd_data["macd"]
        data["macd_signal"] = macd_data["macd_signal"]
        data["macd_histogram"] = macd_data["macd_histogram"]

        # Bollinger Bands
        bb_data = calculate_bollinger_bands(data)
        data["bb_upper"] = bb_data["bb_upper"]
        data["bb_middle"] = bb_data["bb_middle"]
        data["bb_lower"] = bb_data["bb_lower"]

        # Moving averages
        data[f"ma_{self.ma_short}"] = data["close"].rolling(window=self.ma_short).mean()
        data[f"ma_{self.ma_long}"] = data["close"].rolling(window=self.ma_long).mean()
        data["ma_20"] = data["close"].rolling(window=20).mean()
        data["ma_50"] = data["close"].rolling(window=50).mean()

        return data

    def _analyze_current_trend(
        self, historical_data: pd.DataFrame, current_date
    ) -> Any:
        """
        Analyze trend at the current point in time - Simplified but effective real-time trend analysis

        Args:
            historical_data: Historical data (up to the current point in time) 
            current_date: Current date

        Returns:
            Trend analysis results at the current point in time
        """
        try:
            # Ensure sufficient data
            if len(historical_data) < 20:
                print(f"⚠️ Insufficient data for trend analysis ({len(historical_data)} < 20)")
                from types import SimpleNamespace

                trend_analysis = SimpleNamespace()
                trend_analysis.dominant_trend = "sideways"
                trend_analysis.complexity_score = 0.5
                trend_analysis.confidence = 0.3
                return trend_analysis

            # Use multi-timeframe analysis
            data = historical_data.copy()

            # Normalize column names (Handle capitalization differences) 
            column_mapping = {}
            for col in data.columns:
                if col.lower() == "close":
                    column_mapping[col] = "close"
                elif col.lower() == "open":
                    column_mapping[col] = "open"
                elif col.lower() == "high":
                    column_mapping[col] = "high"
                elif col.lower() == "low":
                    column_mapping[col] = "low"
                elif col.lower() == "volume":
                    column_mapping[col] = "volume"

            data = data.rename(columns=column_mapping)

            # Ensure availability of close price data
            if "close" not in data.columns:
                if "Close" in data.columns:
                    data["close"] = data["Close"]
                else:
                    raise ValueError("Cannot find price data (close/Close column) ")

            prices = data["close"]

            # Calculate trends across multiple timeframes
            windows = [5, 10, 20]  # Short-term, Medium-term, Long-term
            trends = []
            trend_strengths = []

            for window in windows:
                if len(prices) >= window + 2:
                    # Use linear regression to calculate trend
                    recent_prices = prices.tail(window)
                    x = np.arange(len(recent_prices))
                    y = recent_prices.values

                    if len(y) > 1:
                        slope, _ = np.polyfit(x, y, 1)
                        correlation = np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0

                        # Normalized slope
                        normalized_slope = slope / recent_prices.mean()

                        # Determine trend direction
                        if abs(normalized_slope) < 0.001:  # Almost no trend
                            trend_direction = "sideways"
                        elif normalized_slope > 0:
                            trend_direction = "uptrend"
                        else:
                            trend_direction = "downtrend"

                        trends.append(trend_direction)
                        trend_strengths.append(abs(correlation))

            # Determine dominant trend
            if not trends:
                dominant_trend = "sideways"
                confidence = 0.3
            else:
                # Count occurrences and strength of each trend
                trend_counts = {"uptrend": 0, "downtrend": 0, "sideways": 0}
                weighted_scores = {"uptrend": 0.0, "downtrend": 0.0, "sideways": 0.0}

                for trend, strength in zip(trends, trend_strengths):
                    trend_counts[trend] += 1
                    weighted_scores[trend] += strength

                # Find the trend with the highest weighted score
                dominant_trend = max(weighted_scores, key=weighted_scores.get)

                # Calculate confidence
                total_strength = sum(trend_strengths)
                if total_strength > 0:
                    confidence = weighted_scores[dominant_trend] / total_strength
                    confidence = min(confidence, 1.0)
                else:
                    confidence = 0.3

            # Check price momentum to confirm trend
            trend_reversal_detected = False
            reversal_strength = 0.0

            if len(prices) >= 15:
                # Detect trend transition signals - Balance upward and downward detection thresholds
                short_term_change = (
                    prices.iloc[-5:].mean() - prices.iloc[-10:-5].mean()
                ) / prices.iloc[-10:-5].mean()
                medium_term_change = (
                    prices.iloc[-10:].mean() - prices.iloc[-20:-10].mean()
                ) / prices.iloc[-20:-10].mean()

                print(
                    f"📊 Transition signal calculation: Short-term change={short_term_change:.4f} ({short_term_change:.2%}), Medium-term change={medium_term_change:.4f} ({medium_term_change:.2%})"
                )

                # Balanced upward and downward transition detection, Use the same threshold
                reversal_threshold = 0.02  # Consistently use 2%threshold
                counter_threshold = 0.01  # Consistently use 1%Reverse threshold

                if (
                    short_term_change > reversal_threshold
                    and medium_term_change < -counter_threshold
                ):
                    trend_reversal_detected = True
                    reversal_strength = abs(short_term_change)
                    print(
                        f"🔄 Upward transition signal detected: Short-term change {short_term_change:.2%}, Medium-term change {medium_term_change:.2%} -> Transition strength {reversal_strength:.2%}"
                    )
                elif (
                    short_term_change < -reversal_threshold
                    and medium_term_change > counter_threshold
                ):
                    trend_reversal_detected = True
                    reversal_strength = abs(short_term_change)
                    print(
                        f"🔄 Downward transition signal detected: Short-term change {short_term_change:.2%}, Medium-term change {medium_term_change:.2%} -> Transition strength {reversal_strength:.2%}"
                    )

                # Additional detection: If the current trend differs from the previous period
                if len(prices) >= 25:
                    very_recent = prices.iloc[-5:].mean()
                    recent = prices.iloc[-10:-5].mean()
                    older = prices.iloc[-15:-10].mean()
                    much_older = prices.iloc[-25:-15].mean()

                    recent_trend = (very_recent - recent) / recent
                    older_trend = (older - much_older) / much_older

                    # Use the same threshold to detect trend changes in both directions
                    trend_change_threshold = 0.02  # Consistent threshold
                    counter_trend_threshold = 0.015  # Consistent reverse threshold

                    if (
                        recent_trend > trend_change_threshold
                        and older_trend < -counter_trend_threshold
                    ):
                        trend_reversal_detected = True
                        reversal_strength = max(reversal_strength, abs(recent_trend))
                        print(
                            f"🔄 Trend direction change detected(Upward): Recent{recent_trend:.2%} vs Earlier{older_trend:.2%} -> Transition strength {reversal_strength:.2%}"
                        )
                    elif (
                        recent_trend < -trend_change_threshold
                        and older_trend > counter_trend_threshold
                    ):
                        trend_reversal_detected = True
                        reversal_strength = max(reversal_strength, abs(recent_trend))
                        print(
                            f"🔄 Trend direction change detected(Downward): Recent{recent_trend:.2%} vs Earlier{older_trend:.2%} -> Transition strength {reversal_strength:.2%}"
                        )

            # Price momentum check - Do not forcibly override the trend, Use only as confirmation
            momentum_factor = 1.0
            if len(prices) >= 10:
                recent_change = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
                momentum_trend = "uptrend" if recent_change > 0 else "downtrend"

                # Momentum confirmation logic - Adjust confidence instead of forcing a trend change
                if abs(recent_change) > 0.08:  # Raise threshold to 8%, Reduce misclassification
                    if recent_change > 0 and dominant_trend == "uptrend":
                        print(f"🔄 Price momentum confirms uptrend (Change: {recent_change:.2%})")
                        confidence = min(confidence + 0.15, 1.0)
                        momentum_factor = 1.2
                    elif recent_change < 0 and dominant_trend == "downtrend":
                        print(f"🔄 Price momentum confirms downtrend (Change: {recent_change:.2%})")
                        confidence = min(confidence + 0.15, 1.0)
                        momentum_factor = 1.2
                    elif abs(recent_change) > 0.12:  # Consider overriding the original trend only with extremely strong momentum
                        if recent_change > 0 and dominant_trend == "downtrend":
                            print(
                                f"� Extremely strong upward momentum overrides downtrend (Change: {recent_change:.2%})"
                            )
                            dominant_trend = "uptrend"
                            confidence = 0.7
                        elif recent_change < 0 and dominant_trend == "uptrend":
                            print(
                                f"� Extremely strong downward momentum overrides uptrend (Change: {recent_change:.2%})"
                            )
                            dominant_trend = "downtrend"
                            confidence = 0.7

            # Calculate complexity score
            unique_trends = len(set(trends))
            complexity_score = unique_trends / len(windows) if windows else 0.5

            # Create trend analysis result
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = dominant_trend
            trend_analysis.confidence = confidence
            trend_analysis.complexity_score = complexity_score

            # Add trend transition information
            trend_analysis.trend_reversal_detected = trend_reversal_detected
            trend_analysis.reversal_strength = reversal_strength

            print(
                f"🎯 Real-time trend analysis: {dominant_trend} (Confidence: {confidence:.2f}, Complexity: {complexity_score:.2f})"
            )
            if trend_reversal_detected:
                print(f"⚡ Trend transition detection: Strength {reversal_strength:.2%}")

            return trend_analysis

        except Exception as e:
            print(f"❌ Real-time trend analysis failed: {e}")
            import traceback

            traceback.print_exc()

            # Return fallback result
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = "sideways"
            trend_analysis.complexity_score = 0.5
            trend_analysis.confidence = 0.2
            return trend_analysis

    def _detect_trigger_events(
        self, data: pd.DataFrame, current_index: int
    ) -> List[Dict[str, Any]]:
        """
        Detect trigger events

        Args:
            data: Historical data
            current_index: Current data index

        Returns:
            List of trigger events
        """
        events = []
        current = data.iloc[current_index]
        prev = data.iloc[current_index - 1] if current_index > 0 else current

        # MACD Trigger events
        if (
            current["macd"] > current["macd_signal"]
            and prev["macd"] <= prev["macd_signal"]
        ):
            events.append(
                {
                    "event_type": "MACD_GOLDEN_CROSS",
                    "severity": "high"
                    if current["macd_histogram"] > 0.01
                    else "medium",
                    "description": "MACD Bullish crossover signal",
                    "technical_data": {
                        "indicator": "MACD_GOLDEN_CROSS",
                        "value": None,
                        "threshold": None,
                        "strength": "high"
                        if current["macd_histogram"] > 0.01
                        else "medium",
                    },
                }
            )
        elif (
            current["macd"] < current["macd_signal"]
            and prev["macd"] >= prev["macd_signal"]
        ):
            events.append(
                {
                    "event_type": "MACD_DEATH_CROSS",
                    "severity": "high"
                    if current["macd_histogram"] < -0.01
                    else "medium",
                    "description": "MACD Bearish crossover signal",
                    "technical_data": {
                        "indicator": "MACD_DEATH_CROSS",
                        "value": None,
                        "threshold": None,
                        "strength": "high"
                        if current["macd_histogram"] < -0.01
                        else "medium",
                    },
                }
            )

        # Bollinger Band trigger events
        if current["close"] <= current["bb_lower"] and prev["close"] > prev["bb_lower"]:
            events.append(
                {
                    "event_type": "BB_LOWER_TOUCH",
                    "severity": "high",
                    "description": "Price touches lower Bollinger Band",
                    "technical_data": {
                        "indicator": "BB_LOWER_TOUCH",
                        "value": None,
                        "threshold": None,
                        "strength": "high",
                    },
                }
            )
        elif (
            current["close"] >= current["bb_upper"] and prev["close"] < prev["bb_upper"]
        ):
            events.append(
                {
                    "event_type": "BB_UPPER_TOUCH",
                    "severity": "high",
                    "description": "Price touches upper Bollinger Band",
                    "technical_data": {
                        "indicator": "BB_UPPER_TOUCH",
                        "value": None,
                        "threshold": None,
                        "strength": "high",
                    },
                }
            )

        # Volume analysis events
        if len(data) >= 10:
            recent_volume = data["volume"].tail(10).mean()
            if current.get("volume", 0) > recent_volume * 2:
                events.append(
                    {
                        "event_type": "VOLUME_SPIKE",
                        "severity": "medium",
                        "description": f"Volume surge ({current.get('volume', 0) / recent_volume:.1f} times)",
                        "technical_data": {
                            "indicator": "VOLUME_SPIKE",
                            "current_volume": int(current.get("volume", 0)),
                            "avg_volume": int(recent_volume),
                            "ratio": float(current.get("volume", 0) / recent_volume),
                            "strength": "high"
                            if current.get("volume", 0) > recent_volume * 3
                            else "medium",
                        },
                    }
                )

        # Price breakout detection
        if len(data) >= 20:
            high_20 = data["high"].tail(20).max()
            low_20 = data["low"].tail(20).min()

            if current["close"] > high_20 and prev["close"] <= high_20:
                events.append(
                    {
                        "event_type": "PRICE_BREAKOUT_HIGH",
                        "severity": "high",
                        "description": f"Break above 20 day high ({high_20:.2f})",
                        "technical_data": {
                            "indicator": "PRICE_BREAKOUT_HIGH",
                            "breakout_level": float(high_20),
                            "current_price": float(current["close"]),
                            "strength": "high",
                        },
                    }
                )
            elif current["close"] < low_20 and prev["close"] >= low_20:
                events.append(
                    {
                        "event_type": "PRICE_BREAKDOWN_LOW",
                        "severity": "high",
                        "description": f"Break below 20 day low ({low_20:.2f})",
                        "technical_data": {
                            "indicator": "PRICE_BREAKDOWN_LOW",
                            "breakdown_level": float(low_20),
                            "current_price": float(current["close"]),
                            "strength": "high",
                        },
                    }
                )

        # Trend reversal events (Use configured moving-average parameters) 
        ma_short_key = f"ma_{self.ma_short}"
        ma_long_key = f"ma_{self.ma_long}"

        if (
            ma_short_key in current
            and ma_long_key in current
            and ma_short_key in prev
            and ma_long_key in prev
        ):
            if (
                current[ma_short_key] > current[ma_long_key]
                and prev[ma_short_key] <= prev[ma_long_key]
            ):
                events.append(
                    {
                        "event_type": "MA_GOLDEN_CROSS",
                        "severity": "medium",
                        "description": f"Short-term moving average({self.ma_short})Cross above long-term moving average({self.ma_long})",
                        "technical_data": {
                            "indicator": "MA_GOLDEN_CROSS",
                            "ma_short": float(current[ma_short_key]),
                            "ma_long": float(current[ma_long_key]),
                            "strength": "medium",
                        },
                    }
                )
            elif (
                current[ma_short_key] < current[ma_long_key]
                and prev[ma_short_key] >= prev[ma_long_key]
            ):
                events.append(
                    {
                        "event_type": "MA_DEATH_CROSS",
                        "severity": "medium",
                        "description": f"Short-term moving average({self.ma_short})Cross below long-term moving average({self.ma_long})",
                        "technical_data": {
                            "indicator": "MA_DEATH_CROSS",
                            "ma_short": float(current[ma_short_key]),
                            "ma_long": float(current[ma_long_key]),
                            "strength": "medium",
                        },
                    }
                )

        # Long bearish K candle detection (Single-day decline 8%or more) 
        if prev["close"] > 0:  # Avoid division by zero
            daily_return = (current["close"] - prev["close"]) / prev["close"]
            if daily_return <= -0.08:  # Decline 8%or more
                events.append(
                    {
                        "event_type": "LARGE_DROP",
                        "severity": "high",
                        "description": f"Long bearish K candle: Single-day decline{daily_return * 100:.2f}%",
                        "technical_data": {
                            "indicator": "LARGE_DROP",
                            "daily_return": float(daily_return),
                            "magnitude": float(abs(daily_return)),
                            "strength": "high",
                        },
                    }
                )
            elif daily_return >= 0.08:  # Rise 8%or more
                events.append(
                    {
                        "event_type": "LARGE_GAIN",
                        "severity": "high",
                        "description": f"Long bullish K candle: Single-day gain{daily_return * 100:.2f}%",
                        "technical_data": {
                            "indicator": "LARGE_GAIN",
                            "daily_return": float(daily_return),
                            "magnitude": float(daily_return),
                            "strength": "high",
                        },
                    }
                )

        # Retain the original 20/50 day moving-average crossover detection
        if current["ma_20"] > current["ma_50"] and prev["ma_20"] <= prev["ma_50"]:
            events.append(
                {
                    "event_type": "TREND_TURN_BULLISH",
                    "severity": "medium",
                    "description": "20 day moving average crosses above 50 day moving average",
                    "technical_data": {
                        "indicator": "TREND_TURN_BULLISH",
                        "ma20": float(current["ma_20"]),
                        "ma50": float(current["ma_50"]),
                        "strength": "medium",
                    },
                }
            )
        elif current["ma_20"] < current["ma_50"] and prev["ma_20"] >= prev["ma_50"]:
            events.append(
                {
                    "event_type": "TREND_TURN_BEARISH",
                    "severity": "medium",
                    "description": "20 day moving average crosses below 50 day moving average",
                    "technical_data": {
                        "indicator": "TREND_TURN_BEARISH",
                        "ma20": float(current["ma_20"]),
                        "ma50": float(current["ma_50"]),
                        "strength": "medium",
                    },
                }
            )

        return events

    def _filter_relevant_events(
        self, events: List[Dict[str, Any]], current_position: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter relevant events by position state, Reduce unnecessary LLM calls

        Args:
            events: All detected events
            current_position: Current position state ('long' or None)

        Returns:
            List of events relevant to the current position
        """
        if not events:
            return []

        # If no position is held, Focus on events related to buy signals (Relax filtering) 
        if not current_position:
            buy_events = [
                "MACD_GOLDEN_CROSS",
                "RSI_OVERSOLD",
                "BB_LOWER_TOUCH",
                "PRICE_ABOVE_MA20",
                "VOLUME_SPIKE",
                "BULLISH_DIVERGENCE",
                "MA_GOLDEN_CROSS",
                "TREND_TURN_BULLISH",  # Add more buy-related events
            ]
            filtered = [
                event
                for event in events
                if any(buy_event in str(event) for buy_event in buy_events)
            ]
            # If no events pass the filter, Retain all important events to avoid omissions
            if (
                not filtered and len(events) <= 3
            ):  # If few events exist and none pass the filter, Retain original events
                return events
            return filtered

        # If a position is held, Focus on events related to sell signals (Relax filtering) 
        else:
            sell_events = [
                "MACD_DEATH_CROSS",
                "RSI_OVERBOUGHT",
                "BB_UPPER_TOUCH",
                "PRICE_BELOW_MA20",
                "BEARISH_DIVERGENCE",
                "VOLUME_DECLINE",
                "MA_DEATH_CROSS",
                "TREND_TURN_BEARISH",
                "LARGE_DROP",  # Add more sell-related events
            ]
            filtered = [
                event
                for event in events
                if any(sell_event in str(event) for sell_event in sell_events)
            ]
            # If no events pass the filter, Retain all important events to avoid omissions
            if (
                not filtered and len(events) <= 3
            ):  # If few events exist and none pass the filter, Retain original events
                return events
            return filtered

    def set_current_symbol(self, symbol: str) -> None:
        """Set the current trading symbol"""
        self.current_symbol = symbol
        print(f"📊 Set trading symbol: {symbol}")

    def finalize_backtest(
        self, final_price: float, final_timestamp: pd.Timestamp
    ) -> None:
        """
        Force settlement of all positions at backtest end

        Args:
            final_price: Closing price of the final trading day
            final_timestamp: Timestamp of the final trading day
        """
        if self.shares > 0 and self.current_position:
            print(f"🏁 Backtest end, Force settlement of positions...")
            print(f"💰 Position quantity: {self.shares} shares")
            print(f"📈 Settlement price: ${final_price:.2f}")

            # Calculate realized P&L
            sale_value = self.shares * final_price
            cost_basis = (
                self.shares * self.position_entry_price
                if self.position_entry_price > 0
                else 0
            )
            realized_pnl = sale_value - cost_basis
            realized_return = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            print(f"💵 Settlement amount: ${sale_value:,.0f}")
            if cost_basis > 0:
                print(f"🎯 Cost basis: ${cost_basis:,.0f}")
                print(f"📊 Realized P&L: ${realized_pnl:,.0f} ({realized_return:+.2f}%)")

            # Update cumulative realized P&L and trade statistics
            if cost_basis > 0:
                self.total_realized_pnl += realized_pnl
                self.trade_returns.append(realized_return)  # Record this trade’s return
                self.total_trades += 1
                is_winning_trade = realized_pnl > 0
                if is_winning_trade:
                    self.winning_trades += 1

                # Calculate current win rate
                current_win_rate = (
                    (self.winning_trades / self.total_trades * 100)
                    if self.total_trades > 0
                    else 0.0
                )
                print(f"💰 Cumulative realized P&L: ${self.total_realized_pnl:,.2f}")
                print(
                    f"📊 Trading statistics: Day {self.total_trades} trades completed, Win rate {current_win_rate:.1f}% ({self.winning_trades}/{self.total_trades})"
                )

            # Update cash balance
            self.cash += sale_value

            # Calculate overall backtest statistics
            total_return = (
                ((self.cash - self.initial_capital) / self.initial_capital * 100)
                if self.initial_capital > 0
                else 0
            )
            print(f"\n📊 === Complete backtest statistics ===")
            print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
            print(f"💵 Final capital: ${self.cash:,.2f}")
            print(f"📈 Total return: {total_return:+.2f}%")
            print(f"🎯 Cumulative realized P&L: ${self.total_realized_pnl:,.2f}")
            print(f"📊 Total trades: {self.total_trades}")
            if self.total_trades > 0:
                print(f"✅ Winning trades: {self.winning_trades}")
                print(
                    f"📊 Overall win rate: {self.winning_trades / self.total_trades * 100:.1f}%"
                )
                print(
                    f"💰 Average P&L per trade: ${self.total_realized_pnl / self.total_trades:,.2f}"
                )

            # Clear position
            final_shares = self.shares  # Save share quantity for signal creation
            self.shares = 0
            self.current_position = None

            # Create settlement trade record
            final_signal = TradingSignal(
                timestamp=final_timestamp,
                signal_type=SignalType.SELL,
                price=final_price,
                confidence=1.0,
                reason="Forced settlement at backtest end",
                metadata={"quantity": final_shares},  # Convert quantity Insert into metadata
            )

            # if present P&L tracker, Update final state
            if (
                hasattr(self, "pnl_tracker")
                and self.pnl_tracker
                and hasattr(self, "current_position_id")
                and self.current_position_id is not None
            ):
                try:
                    self.pnl_tracker.close_position(
                        self.current_position_id,
                        final_price,
                        final_timestamp.strftime("%Y-%m-%d"),
                    )
                    self.current_position_id = None  # Clear position ID
                    print(f"📊 P&L Tracker final state updated")
                except Exception as e:
                    print(f"⚠️ Update P&L tracker failed: {e}")

            print(f"✅ Position settlement completed, Cash balance: ${self.cash:,.0f}")

        else:
            print(f"🏁 Backtest end, No positions to settle")
            print(f"💰 Final cash balance: ${self.cash:,.0f}")

    def get_final_portfolio_value(self, final_price: float) -> float:
        """
        Calculate total portfolio value at backtest end

        Args:
            final_price: Closing price of the final trading day

        Returns:
            Total portfolio value (Cash + Position market value) 
        """
        cash_value = self.cash
        position_value = self.shares * final_price if self.shares > 0 else 0
        total_value = cash_value + position_value

        print(f"📊 Final portfolio value:")
        print(f"   💰 Cash: ${cash_value:,.0f}")
        print(
            f"   📈 Position market value: ${position_value:,.0f} ({self.shares} shares × ${final_price:.2f})"
        )
        print(f"   🎯 Total value: ${total_value:,.0f}")

        return total_value

        # Define entry signals (Watch when no position is held) 
        entry_signals = {
            "BB_LOWER_TOUCH",  # Touch lower Bollinger Band - Oversold rebound
            "MACD_GOLDEN_CROSS",  # MACD Bullish crossover - Bullish signal
            "MA_GOLDEN_CROSS",  # Bullish moving-average crossover - Bullish signal
            "TREND_TURN_BULLISH",  # Trend turns bullish - Entry signal
        }

        # Define exit signals (Watch when holding a position) 
        exit_signals = {
            "BB_UPPER_TOUCH",  # Touch upper Bollinger Band - Overbought pullback
            "MACD_DEATH_CROSS",  # MACD Bearish crossover - Bearish signal
            "MA_DEATH_CROSS",  # Bearish moving-average crossover - Bearish signal
            "TREND_TURN_BEARISH",  # Trend turns bearish - Exit signal
            "LARGE_DROP",  # Long bearish K candle - Sharp decline signal
        }

        relevant_events = []

        # Modify: Simplify logic, Consider all important events
        # This allows LLM consideration of both entry and exit opportunities
        print(f"🔍 Event filtering - Consider all important technical signals")

        for event in events:
            event_type = event["event_type"]

            # Retain all important technical signals
            if event_type in entry_signals or event_type in exit_signals:
                relevant_events.append(event)
                signal_category = (
                    "Entry-related" if event_type in entry_signals else "Exit-related"
                )
                print(f"   ✅ {signal_category}: {event_type} - {event['description']}")
            else:
                print(f"   ❌ Noncritical signals: {event_type} - Filtered")

        return relevant_events

    def _make_llm_decision(
        self,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        events: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Let LLM make trading decisions

        Args:
            data: Historical data
            current_date: Current date
            events: Trigger events
            trend_analysis: Trend analysis results

        Returns:
            LLM Decision results
        """
        try:
            print(f"🧠 Start LLM Decision (Event count: {len(events)})...")
            self.total_llm_calls += 1  # Increase LLM Call count

            # A caller cannot expose later observations through working state.
            data = data.loc[data.index <= current_date]
            if data.empty:
                raise ValueError("No observations are available at the decision date.")
            # Prepare context data
            current_data = data.iloc[-1]
            recent_data = data.tail(5)
            print(f"📊 Context data prepared")

            # Generate comprehensive technical analysis context
            current_date_str = current_date.strftime("%Y-%m-%d")
            comprehensive_context = (
                self.enhanced_analyzer.analyze_comprehensive_context(
                    data, current_date_str, lookback_days=10
                )
            )
            print(f"🔬 Comprehensive technical analysis completed")

            # Store comprehensive technical context for logging
            self.current_comprehensive_context = comprehensive_context

            print(f"📊 Prepare to perform LLM analysis...")

            # Calculate position metrics and P&L insights
            position_metrics = None
            pnl_insights = None

            if hasattr(self, "pnl_tracker") and self.pnl_tracker:
                # Use correct column names (lowercase 'close' instead of uppercase 'Close')
                close_price = current_data.get("close", current_data.get("Close", 0))
                position_metrics = self._calculate_position_metrics(
                    close_price, current_date
                )
                pnl_insights = self._generate_pnl_insights(position_metrics)
                print(
                    f"📈 P&L Analysis completed: Position held={position_metrics.get('has_position', False)}"
                )

            # Build LLM prompt
            prompt = self._build_decision_prompt(
                current_data,
                recent_data,
                events,
                trend_analysis,
                self.stock_characteristics,
                position_metrics,
                pnl_insights,
                comprehensive_context,  # Add comprehensive technical analysis context
            )

            # Check prompt whether it is None
            if prompt is None:
                print("❌ Error: LLM Prompt construction failed (Return None)")
                return None

            print(
                f"📝 LLM Prompt construction completed (Length: {len(prompt)} characters)"
            )

            # The controller owns cold evidence, warm episodes and hot state.
            if self.memory is None:
                self.memory = DecisionMemory()
                self.memory.archive_file(
                    os.path.join(
                        os.path.dirname(__file__), "prompt", "traditional_strategy.md"
                    )
                )
            asof = current_date.isoformat()
            # Only observations available at this decision date enter memory.
            available = data.loc[data.index <= current_date]
            for offset in range(0, len(available), 32):
                window = available.iloc[offset : offset + 32]
                # A partially filled final window has its own immutable source.
                source = (
                    "market:"
                    + window.index[0].isoformat()
                    + ":"
                    + window.index[-1].isoformat()
                )
                self.memory.archive(
                    source,
                    window.to_json(orient="split", date_format="iso"),
                    window.index[-1].isoformat(),
                )
            enhanced = getattr(self, "current_enhanced_analysis", None)
            trend_state = {
                key: getattr(enhanced or trend_analysis, key, None)
                for key in (
                    "market_phase",
                    "dominant_trend",
                    "trend_consistency",
                    "reversal_probability",
                    "momentum_status",
                    "risk_level",
                )
            }
            working = json_value(
                {
                    "symbol": self.current_symbol,
                    "asof": asof,
                    "cash": self.cash,
                    "shares": self.shares,
                    "observation_timeframe": "daily bars through " + asof,
                    "macd_cross": comprehensive_context.get("macd_analysis", {}).get("macd_cross", "unknown"),
                    "macd_alignment": (
                        "above" if current_data.get("macd", float("nan")) > current_data.get("macd_signal", float("nan"))
                        else "below" if current_data.get("macd", float("nan")) < current_data.get("macd_signal", float("nan"))
                        else "equal" if current_data.get("macd", float("nan")) == current_data.get("macd_signal", float("nan"))
                        else "unknown"
                    ),
                    "technical_facts": {
                        "rsi": comprehensive_context.get("momentum_indicators", {}).get("rsi"),
                        "adx": comprehensive_context.get("trend_analysis", {}).get("adx_value"),
                        "resistance": comprehensive_context.get("support_resistance", {}).get("nearest_resistance"),
                        "support": comprehensive_context.get("support_resistance", {}).get("nearest_support"),
                    },
                    "position": str(self.current_position),
                    "current": {
                        key: current_data[key]
                        for key in (
                            "close",
                            "volume",
                            "rsi",
                            "macd",
                            "macd_signal",
                            "bb_upper",
                            "bb_lower",
                            "ma_20",
                            "ma_50",
                        )
                        if key in current_data
                    },
                    "events": events,
                    "trend": trend_state,
                    "entry_price": self.position_entry_price,
                    "entry_date": str(self.position_entry_date),
                }
            )
            config = self.llm_client.config
            budget = prompt_budget(config.num_ctx, config.max_tokens)
            prompt, receipt = self.memory.assemble(
                asof=asof,
                instructions=self.strategy_prompt + "\n" + EXPLANATION_INSTRUCTIONS,
                working=working,
                evidence=prompt,
                query=self.owner_question,
                budget=budget - 900,
            )

            # End the prompt with the output task, after retrieved evidence.
            prompt += (
                "\nReturn decision JSON with action, confidence, risk_level, reasoning, expected_outcome, explanation. "
                "Include all explanation sections specified above. State observed facts separately from "
                "interpretation and forecasts; distinguish timeframes. Conditional SELL may follow a future BUY, "
                "but requires position long. Never report an anticipated crossover as observed.\n"
            )
            # Call LLM
            print(f"🤖 Calling LLM...")
            attempts = []
            for attempt in range(2):
                self.memory.archive(f"attempt-prompt:{asof}:{attempt}", prompt, asof)
                response = self.llm_client.invoke(prompt)
                self.memory.archive(f"attempt-response:{asof}:{attempt}", response.content, asof)
                attempts.append(response.content)
                decision = self._parse_llm_response(response.content)
                try:
                    validate_decision(decision, context=working)
                    break
                except DecisionError as exc:
                    if attempt:
                        self.memory.archive(
                            "rejected:" + asof, json.dumps(attempts), asof
                        )
                        raise DecisionError(
                            "Invalid model decision after correction: " + str(exc)
                        ) from exc
                    # A bounded same-stage retry; never manufacture a HOLD fallback.
                    prompt += (
                        "\nCorrection: " + str(exc)[:360] +
                        " Return the complete decision and all explanation sections; retain concrete English reasoning.\n"
                    )
            receipt["attempts"] = len(attempts)
            receipt["final_prompt_sha256"] = digest(prompt)
            receipt["final_prompt_bytes"] = len(prompt.encode())
            self.memory.archive("final-prompt:" + asof, prompt, asof)
            self.memory.archive("response:" + asof, json.dumps(attempts), asof)
            self.memory.remember(asof, decision)

            # Record decision log
            self.decision_log.append(
                {
                    "date": current_date,
                    "events": events,
                    "memory": receipt,
                    "context": working,
                    "decision": decision,
                    "reasoning": format_decision_reasoning(decision) if decision else "",
                }
            )

            return decision

        except (OllamaError, DecisionError, ValueError):
            # Surface a missing model or server to the backtest stream instead
            # of treating unavailable inference as a valid no-trade decision.
            raise
        except Exception as e:
            print(f"❌ LLM decision error: {e}")
            import traceback

            print(f"🔍 Error details: {traceback.format_exc()}")
            return None
    def _build_decision_prompt(
        self,
        current_data: pd.Series,
        recent_data: pd.DataFrame,
        events: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
        stock_characteristics: Dict[str, Any],
        position_metrics: Optional[Dict[str, Any]] = None,
        pnl_insights: Optional[Dict[str, Any]] = None,
        comprehensive_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build LLM Decision prompt"""

        prompt = f"""
You are a professional stock trading strategy analyst. Write all reasoning and expected outcomes in English. Make a trading decision based on the following information: 

## Stock characteristics analysis
- Volatility: {stock_characteristics.get("volatility", 0):.3f}
- Historical calibration trend consistency: {stock_characteristics.get("trend_consistency", 0):.3f}
- MACD Effectiveness: {stock_characteristics.get("macd_effectiveness", 0):.3f}

## Current market data
- Current price: {current_data["close"]:.2f}
- MACD: {current_data.get("macd", 0):.4f}
- MACD Signal line: {current_data.get("macd_signal", 0):.4f}
- Upper Bollinger Band: {current_data.get("bb_upper", 0):.2f}
- Middle Bollinger Band: {current_data.get("bb_middle", 0):.2f}
- Lower Bollinger Band: {current_data.get("bb_lower", 0):.2f}
- {self.ma_short} day moving average: {current_data.get(f"ma_{self.ma_short}", 0):.2f}
- {self.ma_long} day moving average: {current_data.get(f"ma_{self.ma_long}", 0):.2f}
- 20 day moving average: {current_data.get("ma_20", 0):.2f}
- 50 day moving average: {current_data.get("ma_50", 0):.2f}

## Trigger events
"""

        for event in events:
            prompt += f"- {event['event_type']}: {event['description']} (Severity: {event['severity']})\n"

        # Add comprehensive technical analysis context
        if comprehensive_context and not comprehensive_context.get("error"):
            prompt += f"""
## 📊 Comprehensive technical analysis

### 💰 Price action analysis
- Price change: {comprehensive_context.get("price_action", {}).get("price_change_pct", 0):.2f}%
- Candlestick pattern: {comprehensive_context.get("price_action", {}).get("candle_type", "unknown")}
- Body ratio: {comprehensive_context.get("price_action", {}).get("body_ratio", 0):.2f}
- Volume ratio: {comprehensive_context.get("price_action", {}).get("volume_to_avg_ratio", 1):.2f} times
- Gap: {comprehensive_context.get("price_action", {}).get("gap_pct", 0):.2f}%

### 📈 Moving-average analysis
- MA5: ${comprehensive_context.get("moving_averages", {}).get("ma_5", 0):.2f} (Slope: {comprehensive_context.get("moving_averages", {}).get("ma_5_slope", 0):.4f})
- MA10: ${comprehensive_context.get("moving_averages", {}).get("ma_10", 0):.2f} (Slope: {comprehensive_context.get("moving_averages", {}).get("ma_10_slope", 0):.4f})
- MA20: ${comprehensive_context.get("moving_averages", {}).get("ma_20", 0):.2f} (Slope: {comprehensive_context.get("moving_averages", {}).get("ma_20_slope", 0):.4f})
- Moving-average alignment: {comprehensive_context.get("moving_averages", {}).get("ma_alignment", "unknown")}
- Above all moving averages: {comprehensive_context.get("moving_averages", {}).get("above_all_mas", False)}

### 📊 Volume analysis
- Current volume: {comprehensive_context.get("volume_analysis", {}).get("current_volume", 0):,}
- Volume ratio: {comprehensive_context.get("volume_analysis", {}).get("volume_ratio", 1):.2f} times
- Volume trend: {comprehensive_context.get("volume_analysis", {}).get("volume_trend", 0):.2f}
- Volume surge: {comprehensive_context.get("volume_analysis", {}).get("is_high_volume", False)}
- Price-volume confirmation: {comprehensive_context.get("volume_analysis", {}).get("volume_confirmation", False)}

### 🌊 Volatility analysis
- ATR: {comprehensive_context.get("volatility_analysis", {}).get("atr", 0):.2f}
- Annualized volatility: {comprehensive_context.get("volatility_analysis", {}).get("volatility_annualized", 0):.2f}%
- Volatility percentile: {comprehensive_context.get("volatility_analysis", {}).get("volatility_percentile", 50):.1f}%
- High volatility: {comprehensive_context.get("volatility_analysis", {}).get("is_high_volatility", False)}

### ⚡ Momentum indicators
- RSI: {comprehensive_context.get("momentum_indicators", {}).get("rsi", 50):.2f}
- RSI State: {comprehensive_context.get("momentum_indicators", {}).get("rsi_condition", "neutral")}
- 5 day ROC: {comprehensive_context.get("momentum_indicators", {}).get("roc_5_day", 0):.2f}%
- 10 day ROC: {comprehensive_context.get("momentum_indicators", {}).get("roc_10_day", 0):.2f}%
- Momentum strength: {comprehensive_context.get("momentum_indicators", {}).get("momentum_strength", "neutral")}

### 🎯 Support and resistance
- Nearest resistance: ${comprehensive_context.get("support_resistance", {}).get("nearest_resistance", 0):.2f}
- Nearest support: ${comprehensive_context.get("support_resistance", {}).get("nearest_support", 0):.2f}
- Distance to resistance: {comprehensive_context.get("support_resistance", {}).get("resistance_distance_pct", 0):.2f}%
- Distance to support: {comprehensive_context.get("support_resistance", {}).get("support_distance_pct", 0):.2f}%
- Near key level: {comprehensive_context.get("support_resistance", {}).get("near_resistance", False) or comprehensive_context.get("support_resistance", {}).get("near_support", False)}

### 📐 Trend strength analysis
- Trend direction: {comprehensive_context.get("trend_analysis", {}).get("trend_direction", "neutral")}
- Trend strength: {comprehensive_context.get("trend_analysis", {}).get("trend_strength", 0):.3f}
- ADX value: {comprehensive_context.get("trend_analysis", {}).get("adx_value", 0):.2f}
- Strong trend: {comprehensive_context.get("trend_analysis", {}).get("strong_trend", False)}

### 🏮 Market state
- Market regime: {comprehensive_context.get("market_regime", {}).get("market_regime", "unknown")}
- Regime description: {comprehensive_context.get("market_regime", {}).get("regime_description", "Unknown regime")}
- Trending market: {comprehensive_context.get("market_regime", {}).get("is_trending", False)}
- High volatility: {comprehensive_context.get("market_regime", {}).get("is_volatile", False)}

### 🎈 Bollinger Band analysis
- Bollinger position: {comprehensive_context.get("bollinger_analysis", {}).get("bb_position", 0.5):.3f} (0=Lower band, 1=Upper band)
- Band width: {comprehensive_context.get("bollinger_analysis", {}).get("bb_width", 0):.2f}%
- Band squeeze: {comprehensive_context.get("bollinger_analysis", {}).get("is_squeeze", False)}
- Potential breakout: {comprehensive_context.get("bollinger_analysis", {}).get("potential_breakout", False)}

### 📈 MACD analysis
- MACD line: {comprehensive_context.get("macd_analysis", {}).get("macd_line", 0):.4f}
- Signal line: {comprehensive_context.get("macd_analysis", {}).get("signal_line", 0):.4f}
- Histogram: {comprehensive_context.get("macd_analysis", {}).get("histogram", 0):.4f}
- MACD Position: {comprehensive_context.get("macd_analysis", {}).get("macd_position", "neutral")}
- Crossover signal: {comprehensive_context.get("macd_analysis", {}).get("macd_cross", "none")}
"""

        prompt += f"""
## Trend analysis"""

        # Use Enhanced analysis if available, otherwise fallback to original
        if (
            hasattr(self, "current_enhanced_analysis")
            and self.current_enhanced_analysis
        ):
            enhanced = self.current_enhanced_analysis
            prompt += f"""
- Dominant trend: {enhanced.market_phase} (Enhanced analysis)
- Trend consistency: {enhanced.trend_consistency:.3f}
- Reversal probability: {enhanced.reversal_probability:.3f}
- Momentum status: {enhanced.momentum_status}
- Risk level: {enhanced.risk_level}

📊 **Trend interpretation**: 
- using Enhanced Multi-timeframe analysis, market_phase is the primary basis for trend assessment
- {enhanced.market_phase} represents the current dominant market direction
- Trend consistency{enhanced.trend_consistency:.3f} indicates trend agreement across multiple timeframes"""
        else:
            # Fallback to original analysis
            prompt += f"""
- Dominant trend: {trend_analysis.dominant_trend if trend_analysis else "unknown"} (Basic analysis)
- Trend strength: {trend_analysis.complexity_score if trend_analysis else 0:.3f}

📊 **Trend interpretation**: 
- Use basic trend analysis, dominant_trend is the primary basis for trend assessment"""

            # Add trend transition information
            if (
                hasattr(trend_analysis, "trend_reversal_detected")
                and trend_analysis.trend_reversal_detected
            ):
                # Rate importance by transition strength
                if trend_analysis.reversal_strength > 0.05:  # 5% or more
                    importance = "🔥 Strong transition signal"
                elif trend_analysis.reversal_strength > 0.03:  # 3% or more
                    importance = "⚡ Clear transition signal"
                else:
                    importance = "📊 Mild transition signal"

                prompt += f"""
- {importance}: Trend transition point detected (Strength: {trend_analysis.reversal_strength:.2%})
- 🎯 Key timing: This is a potential trend transition point, Historically, signals of this type often indicate important opportunities
- 💡 Strategy reminder: Transition signal strength ≥ 2% actively consider entry when, ≥ 3% act decisively when"""

        prompt += f"""

## Current position state
Position held: {"Position held" if self.current_position else "No position"}"""

        # Add unrealized P&L information
        if position_metrics and position_metrics.get("has_position"):
            prompt += f"""

### 📈 Position details
- Position quantity: {position_metrics["shares"]:,.0f} shares
- Entry price: ${position_metrics["entry_price"]:.2f}
- Current price: ${position_metrics["current_price"]:.2f}
- Position cost: ${position_metrics["cost_basis"]:,.0f}
- Current market value: ${position_metrics["position_value"]:,.0f}

### 💰 Unrealized P&L analysis
- Unrealized P&L: ${position_metrics["unrealized_pnl"]:,.0f}
- Return: {position_metrics["unrealized_pnl_pct"]:+.2f}%
- Holding days: {position_metrics["holding_days"]} days
- Risk level: {position_metrics["risk_level"]}

### 🎯 P&L insights
- P&L signal: {pnl_insights.get("pnl_signal", "neutral") if pnl_insights else "neutral"}
- Risk warning: {pnl_insights.get("risk_warning", "No special risks") if pnl_insights else "No special risks"}
- Suggested action: {pnl_insights.get("suggested_action", "Normal operation") if pnl_insights else "Normal operation"}"""
        else:
            prompt += f"""

### 📈 Position details
- Position held: No position
- Available capital: ${self.cash:,.0f}
- Total assets: ${self.cash:,.0f}

### 🎯 Investment insights
- Suggested action: {pnl_insights.get("suggested_action", "Consider a new position") if pnl_insights else "Consider a new position"}
- Position recommendation: Normal position allocation"""

        # Add dynamically loaded strategy decision principles
        # Policy is inserted as mandatory instructions by the memory controller.

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM Response"""
        try:
            # Try to extract JSON section
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)

            return None

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def _create_signal_from_decision(
        self, decision: Dict[str, Any], timestamp: pd.Timestamp, price: float
    ) -> Optional[TradingSignal]:
        """from LLM Create trading signal from decision"""

        action = execution_action(decision)
        if action not in ["BUY", "SELL"]:
            return None

        # Additional risk checks: Block clearly unfavorable entries
        if (
            action == "BUY"
            and hasattr(self, "_last_trend_analysis")
            and self._last_trend_analysis
        ):
            trend_analysis = self._last_trend_analysis

            # Check 1: Do not enter during a strong downtrend
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_strength")
                and trend_analysis.trend_strength >= 0.8
            ):
                print(
                    f"🚫 Risk control: Reject entry during a strong downtrend (Trend strength: {trend_analysis.trend_strength:.3f})"
                )
                return None

            # Check 2: Do not enter when downtrend consistency is low
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_consistency")
                and trend_analysis.trend_consistency < 0.5
            ):
                print(
                    f"🚫 Risk control: Avoid entry when the downtrend is unclear (Consistency: {trend_analysis.trend_consistency:.3f})"
                )
                return None

            # Check 3: Downtrend with strength above 0.5 when, requires additional confirmation
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_strength")
                and trend_analysis.trend_strength >= 0.5
            ):
                print(
                    f"⚠️ Risk warning: Entering during a moderately strong downtrend, requires great caution (Trend strength: {trend_analysis.trend_strength:.3f})"
                )
                # Reduce confidence
                decision["confidence"] = min(decision.get("confidence", 0.5), 0.75)

        signal_type = SignalType.BUY if action == "BUY" else SignalType.SELL
        confidence = decision.get("confidence", 0.5)
        reasoning = decision.get("reasoning", "")

        # Update position state
        if hasattr(self, "pnl_tracker") and self.pnl_tracker:
            try:
                if action == "BUY" and not self.current_position:
                    # Fixed 1000 shares
                    shares_to_buy = 1000
                    cost = shares_to_buy * price

                    print(f"🎯 Fixed position size: Buy {shares_to_buy} shares")

                    # Ensure sufficient cash
                    if cost <= self.cash:
                        # Add new position to P&L tracker
                        if self.current_symbol:
                            self.current_position_id = self.pnl_tracker.add_position(
                                self.current_symbol,
                                timestamp.strftime("%Y-%m-%d"),
                                price,
                                shares_to_buy,
                                confidence,
                            )

                        # Update internal position state
                        self.current_position = "long"
                        self.position_entry_price = price
                        self.position_entry_date = timestamp
                        self.shares = shares_to_buy
                        self.cash -= cost

                        # Use fixed stop-loss percentage (5%)
                        stop_loss_price = price * 0.95

                        print(
                            f"📈 Position updated: Buy {shares_to_buy} shares, Price ${price:.2f}, Total cost ${cost:,.0f}"
                        )
                        print(f"🛡️ Stop-loss setting: ${stop_loss_price:.2f} (5%Stop loss)")

                        # Immediately send post-trade P&L Update
                        if self.progress_callback:
                            try:
                                # Calculate current index (Assume this is called within the loop) 
                                day_index = getattr(self, "_current_day_index", 0)
                                total_days = getattr(self, "_total_days", 125)
                                self._send_performance_update(
                                    day_index, total_days, price
                                )
                            except Exception as e:
                                print(f"⚠️ After buying P&L Update failed: {e}")
                    else:
                        print(
                            f"⚠️ Insufficient cash, Unable to buy{shares_to_buy} shares (Requires ${cost:,.0f}, Available ${self.cash:,.0f})"
                        )
                        return None  # Do not generate a signal when funds are insufficient

                elif action == "SELL" and self.current_position and self.shares > 0:
                    # Sell all shares
                    proceeds = self.shares * price

                    # Calculate and record realized P&L
                    cost_basis = self.shares * self.position_entry_price
                    realized_pnl = proceeds - cost_basis
                    realized_return = (
                        (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                    )

                    # Update cumulative realized P&L
                    self.total_realized_pnl += realized_pnl
                    self.trade_returns.append(realized_return)  # Record this trade’s return

                    # Update trade statistics (One complete trade: Buy -> Sell)
                    self.total_trades += 1
                    is_winning_trade = realized_pnl > 0
                    if is_winning_trade:
                        self.winning_trades += 1

                    # Calculate current win rate
                    current_win_rate = (
                        (self.winning_trades / self.total_trades * 100)
                        if self.total_trades > 0
                        else 0.0
                    )

                    # Reset position state
                    self.current_position = None
                    self.position_entry_price = 0.0
                    self.position_entry_date = None
                    old_shares = self.shares
                    self.shares = 0
                    self.cash += proceeds

                    # Close P&L position in tracker
                    if (
                        hasattr(self, "pnl_tracker")
                        and self.pnl_tracker
                        and hasattr(self, "current_position_id")
                        and self.current_position_id is not None
                    ):
                        try:
                            self.pnl_tracker.close_position(
                                self.current_position_id,
                                price,
                                timestamp.strftime("%Y-%m-%d"),
                            )
                            self.current_position_id = None  # Clear position ID
                        except Exception as e:
                            print(f"⚠️ P&L Failed to close position in tracker: {e}")

                    print(f"📉 Position cleared: Sell {old_shares} shares, Price ${price:.2f}")
                    print(f"💰 Sale proceeds: ${proceeds:,.2f}")
                    print(f"🎯 Cost basis: ${cost_basis:,.2f}")
                    print(
                        f"📊 Realized P&L: ${realized_pnl:,.2f} ({realized_return:+.2f}%) ({'✅ Profit' if is_winning_trade else '❌ Loss'})"
                    )
                    print(f"💰 Cumulative realized P&L: ${self.total_realized_pnl:,.2f}")
                    print(
                        f"📊 Trading statistics: Day {self.total_trades} trades completed, Win rate {current_win_rate:.1f}% ({self.winning_trades}/{self.total_trades})"
                    )
                    print(f"💵 Current cash balance: ${self.cash:,.2f}")

                    # Immediately send post-trade P&L Update
                    if self.progress_callback:
                        try:
                            # Calculate current index (Assume this is called within the loop) 
                            day_index = getattr(self, "_current_day_index", 0)
                            total_days = getattr(self, "_total_days", 125)
                            self._send_performance_update(day_index, total_days, price)
                        except Exception as e:
                            print(f"⚠️ After selling P&L Update failed: {e}")

            except Exception as e:
                print(f"⚠️ Position state update failed: {e}")

        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            reason=f"LLM Decision: {reasoning}",
            metadata={
                "decision": decision,
                "risk_level": decision.get("risk_level", "medium"),
                "expected_outcome": decision.get("expected_outcome", ""),
                "position_size": getattr(self, "shares", 0),
                "cash_remaining": getattr(self, "cash", 0),
            },
        )

    # Helper methods (Calculate stock characteristics) 
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency"""
        try:
            returns = data["close"].pct_change().dropna()
            if len(returns) == 0:
                return 0.0

            # Calculate fraction of consecutive moves in the same direction
            direction_changes = (returns > 0).astype(int).diff().abs().sum()
            if len(returns) == 0:
                return 0.0

            consistency = 1.0 - (direction_changes / len(returns))
            return max(0.0, min(1.0, consistency))

        except Exception as e:
            print(f"Trend consistency calculation error: {e}")
            return 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (Based on linear regression of price movements) """
        if len(data) < 10:
            return 0.0

        try:
            prices = data["close"].dropna()
            if len(prices) < 2:
                return 0.0

            x = np.arange(len(prices))

            # Calculate linear regression R²value as a trend strength measure
            correlation_matrix = np.corrcoef(x, prices)
            if correlation_matrix.size == 0:
                return 0.0

            correlation = abs(correlation_matrix[0, 1])
            if np.isnan(correlation):
                return 0.0

            return correlation**2  # R²value

        except Exception as e:
            print(f"Trend strength calculation error: {e}")
            return 0.0

    def _calculate_consecutive_move_tendency(self, returns: pd.Series) -> float:
        """Calculate tendency for consecutive moves (Momentum characteristics) """
        if len(returns) < 5:
            return 0.0

        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0

        for ret in returns:
            if ret > 0:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            elif ret < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up = 0
                consecutive_down = 0

        return (
            (max_consecutive_up + max_consecutive_down) / len(returns)
            if len(returns) > 0
            else 0.0
        )

    def _test_ma_crossover_effectiveness(self, data: pd.DataFrame) -> float:
        """Test moving-average crossover effectiveness"""
        if len(data) < 50:
            return 0.5

        # using 10 day and 20 day moving-average test
        ma_short = data["close"].rolling(10).mean()
        ma_long = data["close"].rolling(20).mean()

        successful_signals = 0
        total_signals = 0

        for i in range(21, len(data) - 5):
            # Bullish crossover
            if (
                ma_short.iloc[i] > ma_long.iloc[i]
                and ma_short.iloc[i - 1] <= ma_long.iloc[i - 1]
            ):
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            # Bearish crossover
            elif (
                ma_short.iloc[i] < ma_long.iloc[i]
                and ma_short.iloc[i - 1] >= ma_long.iloc[i - 1]
            ):
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _test_bollinger_bands_effectiveness(self, data: pd.DataFrame) -> float:
        """Test Bollinger Band effectiveness"""
        if len(data) < 40:
            return 0.5

        # Calculate Bollinger Bands
        bb_data = calculate_bollinger_bands(data, window=20, num_std_dev=2)

        successful_signals = 0
        total_signals = 0

        for i in range(21, len(data) - 5):
            current_price = data["close"].iloc[i]
            bb_upper = bb_data["bb_upper"].iloc[i]
            bb_lower = bb_data["bb_lower"].iloc[i]

            # Touch lower band (Oversold) 
            if current_price <= bb_lower:
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            # Touch upper band (Overbought) 
            elif current_price >= bb_upper:
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _analyze_breakout_tendency(self, data: pd.DataFrame) -> float:
        """Analyze breakout tendency"""
        if len(data) < 20:
            return 0.5

        breakouts = 0
        total_opportunities = 0

        # using 20 day highs and lows as breakout references
        rolling_high = data["high"].rolling(20).max()
        rolling_low = data["low"].rolling(20).min()

        for i in range(20, len(data) - 1):
            if data["close"].iloc[i] > rolling_high.iloc[i - 1]:  # Upward breakout
                if data["close"].iloc[i + 1] > data["close"].iloc[i]:  # Continues rising the next day
                    breakouts += 1
                total_opportunities += 1
            elif data["close"].iloc[i] < rolling_low.iloc[i - 1]:  # Downward breakout
                if data["close"].iloc[i + 1] < data["close"].iloc[i]:  # Continues falling the next day
                    breakouts += 1
                total_opportunities += 1

        return breakouts / total_opportunities if total_opportunities > 0 else 0.5

    def _classify_stock_personality(
        self,
        volatility: float,
        trend_consistency: float,
        reversal_frequency: float,
        macd_effectiveness: float,
    ) -> str:
        """Classify stock behavior from characteristic analysis"""

        if volatility > 0.4 and reversal_frequency > 0.1:
            return "High-volatility range-bound"
        elif volatility > 0.4 and trend_consistency > 0.6:
            return "High-volatility trending"
        elif volatility < 0.2 and trend_consistency > 0.7:
            return "Steady trending"
        elif volatility < 0.2 and reversal_frequency > 0.08:
            return "Low-volatility range-bound"
        elif trend_consistency > 0.8:
            return "Strong trending"
        elif reversal_frequency > 0.12:
            return "Frequent reversals"
        elif macd_effectiveness > 0.7:
            return "Technical-indicator sensitive"
        elif 0.2 <= volatility <= 0.35 and 0.4 <= trend_consistency <= 0.7:
            return "Balanced"
        else:
            return "Complex mixed"

    def _calculate_reversal_frequency(self, data: pd.DataFrame) -> float:
        """Calculate reversal frequency"""
        if len(data) < 10:
            return 0.0

        peaks_valleys = 0
        for i in range(1, len(data) - 1):
            if (
                data["close"].iloc[i] > data["close"].iloc[i - 1]
                and data["close"].iloc[i] > data["close"].iloc[i + 1]
            ) or (
                data["close"].iloc[i] < data["close"].iloc[i - 1]
                and data["close"].iloc[i] < data["close"].iloc[i + 1]
            ):
                peaks_valleys += 1

        return peaks_valleys / len(data) if len(data) > 0 else 0.0

    def _test_macd_effectiveness(self, data: pd.DataFrame) -> float:
        """Test MACD indicator effectiveness"""
        if len(data) < 50:
            return 0.5

        macd_data = calculate_macd(data)
        macd = macd_data["macd"]
        signal = macd_data["macd_signal"]

        successful_signals = 0
        total_signals = 0

        for i in range(1, len(macd) - 5):
            if (
                macd.iloc[i] > signal.iloc[i] and macd.iloc[i - 1] <= signal.iloc[i - 1]
            ):  # Bullish crossover
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            elif (
                macd.iloc[i] < signal.iloc[i] and macd.iloc[i - 1] >= signal.iloc[i - 1]
            ):  # Bearish crossover
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _calculate_gap_frequency(self, data: pd.DataFrame) -> float:
        """Calculate gap frequency"""
        if len(data) < 2:
            return 0.0

        gaps = 0
        for i in range(1, len(data)):
            gap_up = data["low"].iloc[i] > data["high"].iloc[i - 1]
            gap_down = data["high"].iloc[i] < data["low"].iloc[i - 1]
            if gap_up or gap_down:
                gaps += 1

        return gaps / len(data) if len(data) > 0 else 0.0

    def _analyze_support_resistance(self, data: pd.DataFrame) -> float:
        """Analyze support and resistance strength"""
        if len(data) < 20:
            return 0.5

        try:
            # Simplified support and resistance analysis
            low_min = data["low"].min()
            high_max = data["high"].max()

            if pd.isna(low_min) or pd.isna(high_max) or low_min >= high_max:
                return 0.5

            price_levels = np.linspace(low_min, high_max, 20)
            level_touches = []

            for level in price_levels:
                touches = 0
                if level != 0:  # Avoid division by zero
                    for _, row in data.iterrows():
                        if (
                            abs(row["low"] - level) / level < 0.02
                            or abs(row["high"] - level) / level < 0.02
                        ):
                            touches += 1
                level_touches.append(touches)

            if not level_touches:
                return 0.5

            max_touches = max(level_touches)
            return min(1.0, max_touches / len(data)) if len(data) > 0 else 0.5

        except Exception as e:
            print(f"Support and resistance analysis error: {e}")
            return 0.5

    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        LLM Smart strategy (LLM Smart Strategy) - Adaptive parameter optimization version
        
        🧠 Intelligent features: 
        • Automatically analyze stock characteristics (3-6 months of historical data) 
        • Based on volatility, trend behavior, and reversal frequency, dynamically optimize technical indicator parameters
        • No manual parameter tuning required, Strategy automatically adapts to different stock behavior
        
        📊 Stock characteristic analysis dimensions: 
        • Volatility analysis: Annualized volatility, Volatility of volatility
        • Trend characteristics: Trend consistency, Trend strength, Consecutive-move tendency
        • Reversal characteristics: Reversal frequency, Breakout tendency
        • Technical indicator responsiveness: MACD, Moving average, Bollinger Band effectiveness test
        • Overall stock behavior classification: High-volatility trending, Steady trending, range-bound types, etc.
        
        ⚙️ Current parameter settings (Based on configuration) : 
        - Confidence threshold: {self.confidence_threshold}
        - Trend lookback period: {self.trend_lookback} days
        - Event trigger threshold: {self.event_threshold}
        - Maximum daily trades: {self.max_daily_trades}
        - Use technical filter: {"Yes" if self.use_technical_filter else "No"}
        
        🔧 Technical indicators (Dynamic optimization) : 
        - MACD Fast line: {self.macd_fast} (Automatically adjust based on trend behavior)
        - MACD Slow line: {self.macd_slow} (Automatically adjust based on trend behavior)
        - Short-term moving average: {self.ma_short} days (Automatically adjust based on reversal frequency)
        - Long-term moving average: {self.ma_long} days (Automatically adjust based on reversal frequency)
        
        🎯 Workflow: 
        1. Analyze stock characteristics in depth, Generate a stock behavior profile
        2. Optimize all technical indicator parameters based on stock behavior
        3. Detect key technical signals through events
        4. LLM Make trading decisions through comprehensive analysis
        5. Strict risk controls and confidence filtering
        
        ✨ Use cases: 
        - Automatic adaptation across stocks
        - Intelligent trading without manual parameter tuning
        - Suitable for stocks with different behavior
        - Short- to medium-term trading strategies
        """

    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        """Return default configuration"""
        return StrategyConfig(
            name="LLM Smart strategy",
            description="Adaptive strategy based on stock characteristics LLM Trading strategy, Automatically optimize technical parameters",
            parameters={
                "confidence_threshold": 0.6,  # Lower threshold to allow more signals through
                "trend_lookback": 20,
                "event_threshold": 0.05,
                "max_daily_trades": 3,
                "use_technical_filter": True,
                "ma_short": 10,  # Baseline value, Adjusted to stock behavior during use
                "ma_long": 20,  # Baseline value, Adjusted to stock behavior during use
            },
            parameter_specs={
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    display_name="LLM Confidence threshold",
                    description="LLM Minimum confidence required for decisions",
                    param_type=ParameterType.FLOAT,
                    default_value=0.6,  # Lower default value
                    min_value=0.3,
                    max_value=0.95,
                    step=0.05,
                ),
                "trend_lookback": ParameterSpec(
                    name="trend_lookback",
                    display_name="Trend lookback period",
                    description="Lookback days for trend analysis",
                    param_type=ParameterType.INTEGER,
                    default_value=20,
                    min_value=10,
                    max_value=50,
                    step=1,
                ),
                "event_threshold": ParameterSpec(
                    name="event_threshold",
                    display_name="Event trigger threshold",
                    description="Trigger sensitivity for key events",
                    param_type=ParameterType.FLOAT,
                    default_value=0.05,
                    min_value=0.01,
                    max_value=0.2,
                    step=0.01,
                ),
                "max_daily_trades": ParameterSpec(
                    name="max_daily_trades",
                    display_name="Maximum daily trades",
                    description="Maximum trades allowed per day",
                    param_type=ParameterType.INTEGER,
                    default_value=3,
                    min_value=1,
                    max_value=10,
                    step=1,
                ),
                "use_technical_filter": ParameterSpec(
                    name="use_technical_filter",
                    display_name="Technical indicator filtering",
                    description="Whether to filter signals using technical indicators",
                    param_type=ParameterType.BOOLEAN,
                    default_value=True,
                ),
                "ma_short": ParameterSpec(
                    name="ma_short",
                    display_name="Short-term moving-average baseline",
                    description="Baseline period for short-term moving average (Automatically adjusted to stock reversal frequency during use) ",
                    param_type=ParameterType.INTEGER,
                    default_value=10,
                    min_value=5,
                    max_value=20,
                    step=1,
                ),
                "ma_long": ParameterSpec(
                    name="ma_long",
                    display_name="Long-term moving-average baseline",
                    description="Baseline period for long-term moving average (Automatically adjusted to stock reversal frequency during use) ",
                    param_type=ParameterType.INTEGER,
                    default_value=20,
                    min_value=15,
                    max_value=50,
                    step=1,
                ),
            },
            risk_level="medium",
            market_type="all",
            strategy_type="ai_adaptive",
            category="intelligent",
        )

    def _log_daily_analysis(
        self,
        timestamp: pd.Timestamp,
        historical_data: pd.DataFrame,
        i: int,
        events: List[Dict[str, Any]],
        relevant_events: List[Dict[str, Any]],
        trend_analysis: Any,
        llm_decision: Dict[str, Any] = None,
        comprehensive_context: Dict[str, Any] = None,  # New parameter
    ):
        """
        Record daily analysis data in logs

        Args:
            timestamp: Current timestamp
            historical_data: Historical data
            i: Current data index
            events: All detected events
            relevant_events: Relevant events
            trend_analysis: Trend analysis results
            llm_decision: LLM Decision results
            comprehensive_context: Comprehensive technical analysis context
        """
        try:
            current_row = historical_data.iloc[i]
            current_date = timestamp.strftime("%Y-%m-%d")

            # Prepare market data
            market_data = {
                "price": float(current_row.get("close", current_row.get("Close", 0))),
                "volume": int(current_row.get("volume", current_row.get("Volume", 0))),
                "high": float(current_row.get("high", current_row.get("High", 0))),
                "low": float(current_row.get("low", current_row.get("Low", 0))),
                "open": float(current_row.get("open", current_row.get("Open", 0))),
            }

            # Calculate daily return
            if i > 0:
                prev_close = historical_data.iloc[i - 1].get(
                    "close",
                    historical_data.iloc[i - 1].get("Close", market_data["price"]),
                )
                market_data["daily_return"] = (
                    market_data["price"] - prev_close
                ) / prev_close
            else:
                market_data["daily_return"] = 0.0

            # Calculate volatility (Using the previous 10 days of standard deviation) 
            if i >= 10:
                recent_returns = []
                for j in range(max(0, i - 9), i + 1):
                    if j > 0:
                        curr_price = historical_data.iloc[j].get(
                            "close", historical_data.iloc[j].get("Close", 0)
                        )
                        prev_price = historical_data.iloc[j - 1].get(
                            "close",
                            historical_data.iloc[j - 1].get("Close", curr_price),
                        )
                        if prev_price > 0:
                            daily_ret = (curr_price - prev_price) / prev_price
                            recent_returns.append(daily_ret)

                if recent_returns:
                    import numpy as np

                    market_data["volatility"] = float(np.std(recent_returns))
                else:
                    market_data["volatility"] = 0.0
            else:
                market_data["volatility"] = 0.0

            # Prepare trend analysis data
            trend_data = None
            if trend_analysis:
                trend_data = {
                    "short_term": getattr(
                        trend_analysis, "short_term_trend", "neutral"
                    ),
                    "medium_term": getattr(
                        trend_analysis, "medium_term_trend", "neutral"
                    ),
                    "long_term": getattr(trend_analysis, "dominant_trend", "neutral"),
                    "trend_strength": getattr(trend_analysis, "trend_strength", 0.5),
                    "confidence": getattr(trend_analysis, "confidence", 0.5),
                }

                # Add support and resistance level information
                if hasattr(trend_analysis, "support_resistance"):
                    sr = trend_analysis.support_resistance
                    trend_data["support_level"] = getattr(sr, "support", None)
                    trend_data["resistance_level"] = getattr(sr, "resistance", None)

            # Prepare event data
            triggered_events_data = []
            for event in events:
                event_data = {
                    "event_type": event.get("type", "unknown"),
                    "severity": self._determine_event_severity(event),
                    "description": event.get(
                        "description", f"{event.get('type', 'unknown')} Event"
                    ),
                    "technical_data": {
                        "indicator": event.get("indicator", event.get("type")),
                        "value": event.get("value"),
                        "threshold": event.get("threshold"),
                        "strength": event.get("strength", "medium"),
                    },
                }
                triggered_events_data.append(event_data)

            # Prepare LLM Decision data
            llm_decision_data = None
            if llm_decision:
                llm_decision_data = {
                    "decision_made": True,
                    "prompt_version": self.strategy_type,
                    "decision_type": llm_decision.get("action", "HOLD"),
                    "confidence": llm_decision.get("confidence", 0.0),
                    "reasoning": format_decision_reasoning(llm_decision),
                    "explanation": llm_decision.get("explanation"),
                    "key_factors": llm_decision.get("factors", []),
                    "raw_response": llm_decision.get("raw_response", ""),
                }
            else:
                llm_decision_data = {
                    "decision_made": False,
                    "reason": "No significant events or filtered out",
                }

            # Prepare strategy state data
            strategy_state_data = {
                "position": "long" if self.current_position else "neutral",
                "cash": self.cash,
                "portfolio_value": self.current_portfolio_value,
                "shares": self.shares,
                "entry_price": self.position_entry_price
                if self.current_position
                else None,
                "trade_count_today": self.daily_trade_count,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
            }

            # Calculate current P&L
            if self.current_position and self.shares > 0:
                current_value = self.shares * market_data["price"]
                entry_value = self.shares * self.position_entry_price
                strategy_state_data["unrealized_pnl"] = current_value - entry_value
                strategy_state_data["unrealized_pnl_pct"] = (
                    current_value - entry_value
                ) / entry_value
            else:
                strategy_state_data["unrealized_pnl"] = 0.0
                strategy_state_data["unrealized_pnl_pct"] = 0.0

            # Write to log
            log_id = self.backtest_logger.log_daily_analysis(
                symbol=self.current_symbol or "UNKNOWN",
                date=current_date,
                market_data=market_data,
                trend_analysis=trend_data,
                comprehensive_technical_analysis=comprehensive_context,  # New parameter
                triggered_events=triggered_events_data,
                llm_decision=llm_decision_data,
                trading_signal=None,  # Updated separately when signals are generated
                strategy_state=strategy_state_data,
            )

            # Record individual event analysis
            for event in events:
                if event.get("type"):  # Ensure each event has a type
                    self.backtest_logger.log_event_analysis(
                        daily_log_id=log_id,
                        event_type=event.get("type"),
                        severity=self._determine_event_severity(event),
                        market_context={
                            "price_before": market_data["price"],
                            "volume": market_data["volume"],
                            "trend": trend_data.get("short_term", "neutral")
                            if trend_data
                            else "neutral",
                        },
                        llm_response={
                            "triggered_decision": llm_decision is not None,
                            "action_taken": llm_decision.get("action", "HOLD")
                            if llm_decision
                            else "NONE",
                            "confidence": llm_decision.get("confidence", 0.0)
                            if llm_decision
                            else 0.0,
                        },
                    )

            logger.debug(f"✅ Recorded {current_date} analysis data (log_id: {log_id})")

        except Exception as e:
            logger.error(f"❌ Logging failed: {e}")
            import traceback

            traceback.print_exc()

    def _determine_event_severity(self, event: Dict[str, Any]) -> str:
        """
        Determine event severity

        Args:
            event: Event dictionary

        Returns:
            Severity: 'high', 'medium', 'low'
        """
        event_type = event.get("type", "").lower()
        strength = event.get("strength", "medium").lower()

        # Determine severity from event type and strength
        if strength == "high" or event_type in [
            "large_drop",
            "large_gain",
            "volume_spike",
        ]:
            return "high"
        elif strength == "low" or event_type in ["minor_support", "minor_resistance"]:
            return "low"
        else:
            return "medium"

    def _log_trading_signal(
        self,
        timestamp: pd.Timestamp,
        signal: "TradingSignal",
        llm_decision: Dict[str, Any],
    ):
        """
        Log trading signal

        Args:
            timestamp: Signal timestamp
            signal: Trading signal object
            llm_decision: LLM Decision results
        """
        try:
            current_date = timestamp.strftime("%Y-%m-%d")

            # Find log records for the current day
            recent_logs = self.backtest_logger.query_logs(
                symbol=self.current_symbol,
                date_from=current_date,
                date_to=current_date,
                limit=1,
            )

            if recent_logs:
                log_id = recent_logs[0]["id"]

                # Prepare trading signal data
                signal_data = {
                    "signal_type": signal.signal_type.name,
                    "price": signal.price,
                    "quantity": signal.quantity,
                    "confidence": signal.confidence,
                    "reasoning": signal.reason,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "timestamp": timestamp.isoformat(),
                    "llm_factors": llm_decision.get("factors", []),
                    "llm_confidence": llm_decision.get("confidence", 0.0),
                }

                # Update the current day’s record
                with sqlite3.connect(self.backtest_logger.db_path) as conn:
                    conn.execute(
                        """
                        UPDATE daily_analysis_logs 
                        SET trading_signal = ?
                        WHERE id = ?
                    """,
                        (json.dumps(signal_data), log_id),
                    )

                logger.debug(f"✅ Trading signal log updated (log_id: {log_id})")

        except Exception as e:
            logger.error(f"❌ Failed to log trading signal: {e}")

    def get_backtest_summary(self) -> Dict[str, Any]:
        """
        Get backtest summary

        Returns:
            Backtest summary data
        """
        if not self.backtest_logger:
            return {}

        return self.backtest_logger.get_session_summary()

    def export_backtest_logs(self, filepath: str = None):
        """
        Export backtest logs

        Args:
            filepath: Export file path, Use default path if not provided
        """
        if not self.backtest_logger:
            logger.warning("Logger is not enabled, Unable to export")
            return

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"backtest_logs_{self.current_symbol}_{timestamp}.json"

        self.backtest_logger.export_to_json(filepath)
        return filepath
