"""
Custom backtesting engine module

Custom backtesting engine, Provides transparent, and understandable trading logic
Matches typical investors’ trading practices and understanding
Designed as an LLM accessible tool, Provides a concise API interface
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..llm.strategies import SignalType, TradingSignal, TradingStrategy
from ..utils.fetcher import StockDataFetcher
from ..utils.indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
)
from .gates import (
    GateCancelled,
    Stage,
    StageGateController,
    frame_records,
    records_frame,
)

logger = logging.getLogger(__name__)


def strategy_state(strategy) -> dict:
    """Snapshot only the state required by settlement and performance review."""
    return {
        key: getattr(strategy, key)
        for key in (
            "cash",
            "shares",
            "current_position",
            "position_entry_price",
            "position_entry_date",
            "initial_capital",
            "total_trades",
            "winning_trades",
            "total_realized_pnl",
            "trade_returns",
        )
        if hasattr(strategy, key)
    }


class TradingMode(Enum):
    """Trading mode enumeration"""

    LONG_ONLY = "long_only"  # Long only
    SHORT_ONLY = "short_only"  # Short only
    LONG_SHORT = "long_short"  # Long and short


class OrderType(Enum):
    """Order type enumeration"""

    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """Trade status enumeration"""

    PENDING = "pending"  # Pending
    EXECUTED = "executed"  # Executed
    CANCELLED = "cancelled"  # Cancelled
    FAILED = "failed"  # Execution failed


@dataclass
class Trade:
    """
    Trade record - Records complete information for a single trade
    """

    trade_id: str  # Trade ID
    timestamp: datetime  # Trade time
    symbol: str  # Stock symbol
    order_type: OrderType  # Buy or sell
    shares: int  # Shares traded
    price: float  # Trade price
    commission: float  # Commission
    total_cost: float  # Total cost (including commission)
    status: TradeStatus  # Trade status
    signal_confidence: float = 0.0  # Signal strength
    reason: str = ""  # Trade reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format, Ensure all values are JSON serializable"""
        return {
            "trade_id": str(self.trade_id),
            "timestamp": self.timestamp.isoformat()
            if hasattr(self.timestamp, "isoformat")
            else str(self.timestamp),
            "symbol": str(self.symbol),
            "order_type": self.order_type.value,
            "shares": int(self.shares),
            "price": float(self.price),
            "total_cost": float(self.total_cost),
            "commission": float(self.commission)
            if self.commission is not None
            else 0.0,
            "status": self.status.value,
            "signal_confidence": float(self.signal_confidence),
            "reason": str(self.reason) if self.reason else "",
        }


@dataclass
class Portfolio:
    """
    Portfolio state - Track cash and share holdings
    """

    cash: float = 0.0  # Cash balance
    positions: Dict[str, int] = None  # Shares held {Stock symbol: Shares}

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

    def get_position(self, symbol: str) -> int:
        """Get shares held for a specific stock"""
        return self.positions.get(symbol, 0)

    def update_position(self, symbol: str, shares: int) -> None:
        """Update share holdings"""
        if shares == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = shares

    def calculate_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (Cash + Market value of holdings)"""
        stock_value = sum(
            shares * prices.get(symbol, 0.0)
            for symbol, shares in self.positions.items()
        )
        return self.cash + stock_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format, Suitable for LLM processing"""
        return {"cash": self.cash, "positions": self.positions.copy()}


@dataclass
class BacktestConfig:
    """
    Backtest configuration class - Simplified and LLM accessible design
    Removes margin multipliers and maximum position allocation concepts
    Uses maximum share quantities familiar to ordinary investors instead
    """

    initial_capital: float = 1000000.0  # Initial capital
    max_shares_per_trade: int = 1000  # Maximum shares per buy
    trading_mode: TradingMode = TradingMode.LONG_ONLY  # Trading mode
    trade_on_open: bool = (
        False  # Whether to trade at the opening price (False=Closing price)
    )
    commission_rate: float = (
        0.001425  # Commission rate (Approximately for Taiwan stocks 0.1425%)
    )
    min_commission: float = 20.0  # Minimum commission (Taiwan stocks 20 currency units)

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission"""
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format, Convenient for LLM processing"""
        return {
            "initial_capital": self.initial_capital,
            "max_shares_per_trade": self.max_shares_per_trade,
            "trading_mode": self.trading_mode.value,
            "trade_on_open": self.trade_on_open,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """Create configuration from a dictionary, Convenient for LLM calls"""
        config_data = data.copy()
        if "trading_mode" in config_data:
            config_data["trading_mode"] = TradingMode(config_data["trading_mode"])
        return cls(**config_data)


class CustomBacktestEngine:
    """
    Custom backtest engine - Concise, and transparent trading logic

    Designed for ordinary investors, Avoids complex margin and position allocation concepts
    Provides clear cash flow management and share tracking
    Suitable for LLM Concise interface for tool calls API
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine

        Args:
            config: Backtest settings, When set to None use default settings
        """
        self.config = config or BacktestConfig()
        self.strategies: Dict[str, TradingStrategy] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Backtest state
        self.is_running = False
        self.current_results: Dict[str, Any] = {}

    def add_strategy(self, name: str, strategy: TradingStrategy) -> None:
        """
        Add strategy

        Args:
            name: Strategy name
            strategy: Strategy object
        """
        self.strategies[name] = strategy
        logger.info(f"Add strategy: {name}")

    def load_data(
        self,
        symbol: str,
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load backtest data

        Args:
            symbol: Stock symbol
            period: Data period (1y, 6mo, 3mo etc.)
            start_date: Start date (Optional)
            end_date: End date (Optional)

        Returns:
            Processed stock data
        """
        logger.info(f"Load {symbol} Data, Period: {period}")

        try:
            # Fetch data
            data = StockDataFetcher.fetch_stock_data(symbol, period)

            if data is None or data.empty:
                raise ValueError(f"Unable to retrieve {symbol} data")

            # Normalize column names
            data.columns = data.columns.str.lower()

            # Filter data by analysis period
            if hasattr(data, "attrs") and "analysis_start_date" in data.attrs:
                analysis_start = pd.to_datetime(data.attrs["analysis_start_date"])

                # Handle timezone compatibility
                if data.index.tz is not None and analysis_start.tz is None:
                    analysis_start = analysis_start.tz_localize(data.index.tz)
                elif data.index.tz is None and analysis_start.tz is not None:
                    analysis_start = analysis_start.tz_localize(None)

                logger.info(
                    f"Filter data to the analysis period: from {analysis_start.date()} to {data.index.max().date()}"
                )
                data = data[analysis_start:]

            # If an additional date range was specified, apply additional filtering
            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                data = data[start:end]

            # Ensure correct data types
            for col in ["open", "high", "low", "close", "volume"]:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

            # Remove null values
            data = data.dropna()

            # Cache data
            self.data_cache[symbol] = data

            logger.info(f"Load {len(data)} records {symbol} Data")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

            return data

        except Exception as e:
            logger.error(f"Load {symbol} error while processing data: {e}")
            raise

    def run_backtest(
        self,
        stock_data: pd.DataFrame,
        strategy: TradingStrategy,
        initial_cash: float = 10000.0,
        transaction_cost: float = 0.001,
        symbol: str = None,
        gates: Optional[StageGateController] = None,
        report_builder=None,
    ) -> Dict[str, Any]:
        """
        Run backtest - Core method of the custom engine

        Args:
            stock_data: Stock data DataFrame
            strategy: Strategy object (May be a single or combined strategy)
            initial_cash: Initial capital
            transaction_cost: Transaction cost rate
            symbol: Stock symbol

        Returns:
            Backtest results dictionary
        """
        gates = gates or StageGateController(
            request={
                "symbol": symbol,
                "initial_cash": initial_cash,
                "transaction_cost": transaction_cost,
            }
        )
        self.is_running = True

        try:
            if gates.state["stages"][0]["status"] == "pending":
                gates.start(Stage.ACQUIRE)
                gates.complete(Stage.ACQUIRE, {"rows": frame_records(stock_data)})
            gates.start(Stage.PREPARE)
            stock_data = records_frame(
                gates.store.artifact(gates.run_id, Stage.ACQUIRE)["rows"]
            )
            # Determine strategy name
            strategy_name = getattr(strategy, "name", type(strategy).__name__)

            logger.info(f"Start backtest, Use strategy {strategy_name}")

            # 1. Initialize portfolio state
            portfolio = Portfolio(cash=initial_cash)

            # 2. Calculate technical indicators
            enhanced_data = self._prepare_technical_indicators(stock_data.copy())

            gates.complete(Stage.PREPARE, {"rows": frame_records(enhanced_data)})
            gates.start(Stage.ANALYZE)
            enhanced_data = records_frame(
                gates.store.artifact(gates.run_id, Stage.PREPARE)["rows"]
            )

            # 3. Generate trading signals
            # Set the current stock symbol on the strategy, for Enhanced trend analysis
            if hasattr(strategy, "set_current_symbol"):
                strategy.set_current_symbol(symbol or "UNKNOWN")
            elif hasattr(strategy, "current_symbol"):
                strategy.current_symbol = symbol or "UNKNOWN"

            # Set the strategy’s initial capital, Ensure consistency with portfolio consistent
            if hasattr(strategy, "cash") and hasattr(strategy, "initial_capital"):
                strategy.initial_capital = initial_cash
                strategy.cash = initial_cash
                strategy.current_portfolio_value = initial_cash
                strategy.max_portfolio_value = initial_cash
                print(f"💰 Set initial strategy capital: ${initial_cash:,.0f}")

            analyzer = getattr(strategy, "trend_analyzer", None)
            if analyzer is not None and hasattr(analyzer, "use_snapshot"):
                analyzer.use_snapshot(
                    symbol or "UNKNOWN",
                    [
                        {"date": index.strftime("%Y-%m-%d"), **row.to_dict()}
                        for index, row in stock_data.iterrows()
                    ],
                )
            signals = strategy.generate_signals(enhanced_data)

            gates.complete(
                Stage.ANALYZE,
                {
                    "rows_processed": getattr(
                        strategy, "analysis_rows_processed", len(enhanced_data)
                    ),
                    "timestamps": [index.isoformat() for index in enhanced_data.index],
                    "signals": [signal.to_dict() for signal in signals],
                    "decisions": getattr(strategy, "decision_log", []),
                    "strategy_state": strategy_state(strategy),
                },
            )
            gates.start(Stage.SIMULATE)
            sealed = gates.store.artifact(gates.run_id, Stage.ANALYZE)
            signals = [
                TradingSignal(
                    signal_type=SignalType[row["signal_type"]],
                    timestamp=pd.Timestamp(row["timestamp"]),
                    price=row["price"],
                    quantity=row["quantity"],
                    confidence=row["confidence"],
                    reason=row["reason"],
                    stop_loss=row["stop_loss"],
                    take_profit=row["take_profit"],
                    metadata=row["metadata"],
                )
                for row in sealed["signals"]
            ]
            for key, value in sealed["strategy_state"].items():
                if key == "position_entry_date" and value is not None:
                    value = pd.Timestamp(value)
                setattr(strategy, key, value)
            logger.info(f"Generated {len(signals)} trading signals")

            # 4. Run trading simulation
            trades, portfolio_history = self._simulate_trading(
                signals,
                enhanced_data,
                portfolio,
                symbol=symbol or "UNKNOWN",
                transaction_cost=transaction_cost,
                initial_cash=initial_cash,
            )

            # 4.5. Backtest finalization - Force settlement of positions
            final_date = enhanced_data.index[-1]
            final_price = float(enhanced_data.iloc[-1]["close"])

            # If the strategy has a finalize_backtest method, call it
            if hasattr(strategy, "finalize_backtest"):
                try:
                    strategy.finalize_backtest(final_price, final_date)
                    logger.info(
                        f"Strategy {strategy_name} Backtest finalization completed"
                    )
                except Exception as e:
                    logger.warning(
                        f"Strategy {strategy_name} Backtest finalization failed: {e}"
                    )

            # If the portfolio still has positions, force settlement
            current_position = portfolio.get_position(symbol or "UNKNOWN")
            if current_position > 0:
                logger.info(
                    f"Detected unsettled position {current_position} shares, force settlement"
                )

                # Create forced settlement trade
                final_trade = Trade(
                    trade_id=f"FINAL_{len(trades)}",
                    timestamp=final_date,
                    symbol=symbol or "UNKNOWN",
                    order_type=OrderType.SELL,
                    shares=current_position,
                    price=final_price,
                    commission=0.0,  # No commission charged at backtest end
                    total_cost=current_position * final_price,
                    status=TradeStatus.EXECUTED,
                    signal_confidence=1.0,
                    reason="Forced settlement at backtest end",
                )

                # Execute forced settlement
                portfolio.cash += current_position * final_price
                portfolio.update_position(symbol or "UNKNOWN", 0)  # Clear position
                trades.append(final_trade)

                # Update the final portfolio history record
                if portfolio_history:
                    portfolio_history[-1].update(
                        {
                            "cash": portfolio.cash,
                            "position": 0,
                            "stock_value": 0,
                            "total_value": portfolio.cash,
                            "cumulative_return": (portfolio.cash - initial_cash)
                            / initial_cash,
                        }
                    )

                logger.info(
                    f"Forced settlement completed: Sold {current_position} shares, Price ${final_price:.2f}"
                )

            gates.complete(
                Stage.SIMULATE,
                {
                    "trades": [trade.to_dict() for trade in trades],
                    "portfolio_history": portfolio_history,
                    "strategy_state": strategy_state(strategy),
                },
            )
            gates.start(Stage.REVIEW)
            replay = gates.store.artifact(gates.run_id, Stage.SIMULATE)
            portfolio_history = [
                {**row, "date": pd.Timestamp(row["date"])}
                for row in replay["portfolio_history"]
            ]
            trades = [
                Trade(
                    **{
                        **row,
                        "timestamp": pd.Timestamp(row["timestamp"]),
                        "order_type": OrderType(row["order_type"]),
                        "status": TradeStatus(row["status"]),
                    }
                )
                for row in replay["trades"]
            ]
            enhanced_data = records_frame(
                gates.store.artifact(gates.run_id, Stage.PREPARE)["rows"]
            )

            # 5. Calculate performance metrics
            results = self._calculate_performance_metrics(
                portfolio_history,
                enhanced_data,
                trades,
                signals,
                symbol=symbol or "UNKNOWN",
                strategy_name=strategy_name,
                strategy=SimpleNamespace(**replay["strategy_state"]),
                initial_cash=initial_cash,
            )

            gates.complete(Stage.REVIEW, {"results": results})
            gates.start(Stage.REPORT)
            results = gates.store.artifact(gates.run_id, Stage.REVIEW)["results"]
            # Preserve the engine's datetime-valued ledger for Python callers.
            for row in results["portfolio_history"]:
                row["date"] = pd.Timestamp(row["date"])
            reported = report_builder(results) if report_builder else results
            gates.complete(Stage.REPORT, {"results": results, "view": reported})
            self.last_pipeline = gates.store.snapshot(gates.run_id)

            # 6. Store results
            self.current_results[
                f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ] = results

            logger.info(f"Backtest completed: {strategy_name}")
            return reported

        except GateCancelled:
            raise
        except Exception as e:
            gates.fail(e)
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.is_running = False

    def _prepare_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare technical indicator data

        Args:
            data: Raw stock price data

        Returns:
            Data containing technical indicators
        """
        # Calculate technical indicators
        if "5ma" not in data.columns:
            data = calculate_moving_averages(data)
        if "bb_upper" not in data.columns:
            data = calculate_bollinger_bands(data)
        if "rsi" not in data.columns:
            data = calculate_rsi(data)
        if "macd" not in data.columns:
            data = calculate_macd(data)

        return data

    def _simulate_trading(
        self,
        signals: List[TradingSignal],
        data: pd.DataFrame,
        portfolio: Portfolio,
        symbol: str,
        transaction_cost: float = 0.001,
        initial_cash: float = 10000.0,
    ) -> Tuple[List[Trade], List[Dict[str, Any]]]:
        """
        Simulate trade execution - Core trading logic

        Args:
            signals: List of trading signals
            data: Stock price data
            portfolio: Portfolio state
            symbol: Stock symbol

        Returns:
            (List of trade records, Portfolio history)
        """
        trades = []
        portfolio_history = []
        trade_counter = 0

        # Build signal lookup table
        signal_dict = {}
        for signal in signals:
            date_key = signal.timestamp.date()
            if date_key not in signal_dict:
                signal_dict[date_key] = []
            signal_dict[date_key].append(signal)

        # Simulate trading day by day
        for date, row in data.iterrows():
            current_date = date.date() if hasattr(date, "date") else date
            current_price = float(row["close"])

            # Check for signals on the current day
            daily_signals = signal_dict.get(current_date, [])

            # Process trading signals
            for signal in daily_signals:
                if signal.signal_type == SignalType.BUY:
                    trade = self._execute_buy_order(
                        signal,
                        current_price,
                        portfolio,
                        symbol,
                        trade_counter,
                        transaction_cost,
                    )
                    if trade:
                        trades.append(trade)
                        trade_counter += 1

                elif signal.signal_type == SignalType.SELL:
                    trade = self._execute_sell_order(
                        signal,
                        current_price,
                        portfolio,
                        symbol,
                        trade_counter,
                        transaction_cost,
                    )
                    if trade:
                        trades.append(trade)
                        trade_counter += 1

            # Record daily portfolio state
            current_position = portfolio.get_position(symbol)
            stock_value = current_position * current_price
            total_value = portfolio.cash + stock_value

            # Calculate cumulative return (relative to initial capital)
            cumulative_return = (total_value - initial_cash) / initial_cash

            # Calculate unrealized P&L and return for this trade
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0
            position_entry_price = 0.0

            # If a position is held, Calculate unrealized P&L
            if current_position > 0:
                # Find entry price from the most recent buy trade
                recent_buy_trades = [t for t in trades if t.order_type.value == "buy"]
                if recent_buy_trades:
                    latest_buy_trade = recent_buy_trades[-1]
                    position_entry_price = latest_buy_trade.price
                    cost_basis = current_position * position_entry_price
                    unrealized_pnl = stock_value - cost_basis
                    unrealized_pnl_pct = (
                        (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
                    )

            portfolio_snapshot = {
                "date": date,
                "cash": portfolio.cash,
                "position": current_position,
                "stock_price": current_price,
                "stock_value": stock_value,
                "total_value": total_value,
                "cumulative_return": cumulative_return,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "position_entry_price": position_entry_price,
                "position_cost_basis": current_position * position_entry_price
                if current_position > 0
                else 0.0,
            }
            portfolio_history.append(portfolio_snapshot)

        return trades, portfolio_history

    def _execute_buy_order(
        self,
        signal: TradingSignal,
        price: float,
        portfolio: Portfolio,
        symbol: str,
        trade_id: int,
        transaction_cost: float = 0.001,
    ) -> Optional[Trade]:
        """
        Execute buy order

        Args:
            signal: Buy signal
            price: Execution price
            portfolio: Portfolio state
            symbol: Stock symbol
            trade_id: Trade ID

        Returns:
            Trade record or None (if execution is not possible)
        """
        # In long-only mode, do not buy again if shares are already held
        if (
            self.config.trading_mode == TradingMode.LONG_ONLY
            and portfolio.get_position(symbol) > 0
        ):
            logger.info(f"Already holding {symbol}, Skip buy signal")
            return None

        # Calculate affordable shares
        max_shares = (
            self.config.max_shares_per_trade
            if hasattr(self.config, "max_shares_per_trade")
            else 100
        )
        trade_value = max_shares * price
        commission = trade_value * transaction_cost
        total_cost = trade_value + commission

        # Check available funds
        if portfolio.cash < total_cost:
            # Adjust to affordable share quantity
            available_cash = portfolio.cash - commission
            if available_cash <= 0:
                logger.warning(f"Insufficient funds, Unable to buy {symbol}")
                return None

            max_shares = int(available_cash // price)
            if max_shares <= 0:
                logger.warning(f"Insufficient funds, Unable to buy any {symbol} shares")
                return None

            trade_value = max_shares * price
            commission = trade_value * transaction_cost
            total_cost = trade_value + commission

        # Execute buy
        portfolio.cash -= total_cost
        current_position = portfolio.get_position(symbol)
        portfolio.update_position(symbol, current_position + max_shares)

        # Create trade record
        trade = Trade(
            trade_id=f"T{trade_id:04d}",
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type=OrderType.BUY,
            shares=max_shares,
            price=price,
            commission=commission,
            total_cost=total_cost,
            status=TradeStatus.EXECUTED,
            signal_confidence=signal.confidence,
            reason=signal.reason,
        )

        logger.info(f"Buy executed: {symbol} {max_shares} shares @ ${price:.2f}")
        return trade

    def _execute_sell_order(
        self,
        signal: TradingSignal,
        price: float,
        portfolio: Portfolio,
        symbol: str,
        trade_id: int,
        transaction_cost: float = 0.001,
    ) -> Optional[Trade]:
        """
        Execute sell order

        Args:
            signal: Sell signal
            price: Execution price
            portfolio: Portfolio state
            symbol: Stock symbol
            trade_id: Trade ID

        Returns:
            Trade record or None (if execution is not possible)
        """
        current_position = portfolio.get_position(symbol)

        # Check for shares available to sell
        if current_position <= 0:
            logger.debug(f"No {symbol} holdings, Unable to execute sale")
            return None

        # Sell all shares held
        shares_to_sell = current_position
        trade_value = shares_to_sell * price
        commission = trade_value * transaction_cost
        proceeds = trade_value - commission

        # Execute sell
        portfolio.cash += proceeds
        portfolio.update_position(symbol, 0)

        # Create trade record
        trade = Trade(
            trade_id=f"T{trade_id:04d}",
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type=OrderType.SELL,
            shares=shares_to_sell,
            price=price,
            commission=commission,
            total_cost=proceeds,  # For sell orders, this represents proceeds
            status=TradeStatus.EXECUTED,
            signal_confidence=signal.confidence,
            reason=signal.reason,
        )

        logger.info(f"Sell executed: {symbol} {shares_to_sell} shares @ ${price:.2f}")
        return trade

    def _calculate_performance_metrics(
        self,
        portfolio_history: List[Dict[str, Any]],
        data: pd.DataFrame,
        trades: List[Trade],
        signals: List[TradingSignal],
        symbol: str,
        strategy_name: str,
        strategy: Union[TradingStrategy, None] = None,  # Add strategy object parameter
        initial_cash: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics

        Args:
            portfolio_history: Portfolio history
            data: Stock price data
            trades: Trade record
            signals: Original trading signals
            symbol: Stock symbol
            strategy_name: Strategy name

        Returns:
            Performance metrics dictionary
        """
        if not portfolio_history:
            raise ValueError("No portfolio history data")

        # Basic information
        start_date = portfolio_history[0]["date"]
        end_date = portfolio_history[-1]["date"]
        total_days = len(portfolio_history)

        # Final value
        final_value = portfolio_history[-1]["total_value"]
        final_return = portfolio_history[-1]["cumulative_return"]

        # Calculate annualized return
        days_in_year = 365.25
        years = total_days / days_in_year
        annual_return = (
            (final_value / self.config.initial_capital) ** (1 / years) - 1
            if years > 0
            else 0
        )

        # Calculate volatility
        returns = [ph["cumulative_return"] for ph in portfolio_history]
        returns_series = pd.Series(returns)
        daily_returns = returns_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

        # processing NaN value
        if pd.isna(volatility):
            volatility = 0.0

        # Calculate maximum drawdown
        values = [ph["total_value"] for ph in portfolio_history]
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax
        max_drawdown = drawdown.min()

        # processing NaN value
        if pd.isna(max_drawdown):
            max_drawdown = 0.0

        # Trading statistics
        num_trades = len(trades)
        buy_trades = [t for t in trades if t.order_type == OrderType.BUY]
        sell_trades = [t for t in trades if t.order_type == OrderType.SELL]

        # Calculate win rate (Requires matching buy and sell trades)
        win_rate = 0.0
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Simplified win-rate calculation: Compare buy and sell prices
            paired_trades = min(len(buy_trades), len(sell_trades))
            wins = sum(
                1
                for i in range(paired_trades)
                if sell_trades[i].price > buy_trades[i].price
            )
            win_rate = wins / paired_trades if paired_trades > 0 else 0

        # Benchmark comparison (Buy and hold)
        buy_hold_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1
        alpha = final_return - buy_hold_return

        # Generate trading events
        trading_events = self._generate_trading_events(
            trades, portfolio_history, symbol
        )

        # Convert original signals to frontend-compatible format
        trading_signals = []
        for signal in signals:
            signal_type_str = (
                signal.signal_type.name
                if hasattr(signal.signal_type, "name")
                else str(signal.signal_type)
            )
            trading_signals.append(
                {
                    "timestamp": signal.timestamp.isoformat()
                    if hasattr(signal.timestamp, "isoformat")
                    else str(signal.timestamp),
                    "signal_type": signal_type_str.upper(),
                    "confidence": float(signal.confidence),
                    "price": float(signal.price) if signal.price else 0.0,
                    "reason": str(signal.reason) if signal.reason else "",
                    "metadata": signal.metadata or {},
                }
            )

        # Combine results, Ensure all numbers are Python native types
        results = {
            "basic_info": {
                "symbol": str(symbol),
                "strategy_name": str(strategy_name),
                "start_date": start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date),
                "end_date": end_date.isoformat()
                if hasattr(end_date, "isoformat")
                else str(end_date),
                "total_days": int(total_days),
                "initial_capital": float(initial_cash),
                "max_shares_per_trade": int(self.config.max_shares_per_trade),
            },
            "performance_metrics": {
                "final_value": float(final_value),
                "total_return": float(final_return),
                "annual_return": float(annual_return),
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "num_trades": int(num_trades),
                "win_rate": float(win_rate),
            },
            "strategy_statistics": {
                # Get detailed statistics from the strategy (if available)
                "total_realized_pnl": getattr(strategy, "total_realized_pnl", 0.0),
                "total_trades": getattr(strategy, "total_trades", num_trades),
                "winning_trades": getattr(strategy, "winning_trades", 0),
                "strategy_win_rate": (
                    getattr(strategy, "winning_trades", 0)
                    / getattr(strategy, "total_trades", 1)
                )
                if getattr(strategy, "total_trades", 0) > 0
                else 0.0,
                "cumulative_trade_return_rate": sum(
                    getattr(strategy, "trade_returns", [])
                )
                / 100
                if hasattr(strategy, "trade_returns") and strategy.trade_returns
                else 0.0,
            },
            "benchmark_comparison": {
                "buy_hold_return": float(buy_hold_return),
                "strategy_return": float(final_return),
                "alpha": float(alpha),
                "outperformed": bool(alpha > 0),
            },
            "trades": [trade.to_dict() for trade in trades],
            "trading_signals": trading_signals,  # New: Original trading signals
            "portfolio_history": portfolio_history,
            "trading_events": trading_events,
            "stock_data": [
                {
                    "timestamp": idx.isoformat()
                    if hasattr(idx, "isoformat")
                    else str(idx),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                }
                for idx, row in data.iterrows()
            ],
        }

        return results

    def _generate_trading_events(
        self, trades: List[Trade], portfolio_history: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed trading event logs

        Args:
            trades: List of trade records
            portfolio_history: Portfolio history
            symbol: Stock symbol

        Returns:
            Trading event list
        """
        events = []

        # Map dates to portfolio states
        portfolio_dict = {}
        for ph in portfolio_history:
            date_key = ph["date"].date() if hasattr(ph["date"], "date") else ph["date"]
            portfolio_dict[date_key] = ph

        # Process each trade
        for trade in trades:
            trade_date = (
                trade.timestamp.date()
                if hasattr(trade.timestamp, "date")
                else trade.timestamp
            )
            portfolio_state = portfolio_dict.get(trade_date, {})

            if trade.order_type == OrderType.BUY:
                event_type = "buy_success"
                description = f"Buy signal executed successfully, Buy {symbol} {trade.shares} shares, Price ${trade.price:.2f}"
            else:
                event_type = "sell_success"
                description = f"Sell signal executed successfully, Sell {symbol} {trade.shares} shares, Price ${trade.price:.2f}"

            events.append(
                {
                    "date": trade.timestamp.isoformat()
                    if hasattr(trade.timestamp, "isoformat")
                    else str(trade.timestamp),
                    "event_type": event_type,
                    "signal_type": trade.order_type.value.upper(),
                    "signal_confidence": trade.signal_confidence,
                    "execution_price": trade.price,
                    "shares_traded": trade.shares,
                    "trade_amount": trade.total_cost,
                    "commission": trade.commission,
                    "current_position": portfolio_state.get("position", 0),
                    "current_cash": portfolio_state.get("cash", 0),
                    "current_equity": portfolio_state.get("total_value", 0),
                    "cumulative_return": portfolio_state.get("cumulative_return", 0),
                    "description": description,
                }
            )

        # Add final settlement event
        if portfolio_history:
            final_state = portfolio_history[-1]
            events.append(
                {
                    "date": final_state["date"].isoformat()
                    if hasattr(final_state["date"], "isoformat")
                    else str(final_state["date"]),
                    "event_type": "final_settlement",
                    "signal_type": "SETTLEMENT",
                    "current_position": final_state["position"],
                    "current_cash": final_state["cash"],
                    "stock_value": final_state["stock_value"],
                    "current_equity": final_state["total_value"],
                    "cumulative_return": final_state["cumulative_return"],
                    "description": f"Final settlement - Holding{symbol} {final_state['position']} shares, Cash ${final_state['cash']:,.0f}, Total assets ${final_state['total_value']:,.0f}, Cumulative return {final_state['cumulative_return'] * 100:.2f}%",
                }
            )

        return events

    def get_backtest_chart(
        self, symbol: str, strategy_name: str, show_trades: bool = True
    ) -> go.Figure:
        """
        Generate backtest results chart

        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            show_trades: Whether to display trade markers

        Returns:
            Plotly Chart object
        """
        result_key = f"{symbol}_{strategy_name}"

        if result_key not in self.current_results:
            raise ValueError(
                f"Cannot find {symbol} Strategy {strategy_name} backtest results"
            )

        result = self.current_results[result_key]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f"{symbol} Price and trades", "Cumulative return"],
            row_heights=[0.7, 0.3],
        )

        # Stock price data
        stock_data = result["stock_data"]
        dates = [pd.to_datetime(d["timestamp"]) for d in stock_data]

        # Add stock candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=[d["open"] for d in stock_data],
                high=[d["high"] for d in stock_data],
                low=[d["low"] for d in stock_data],
                close=[d["close"] for d in stock_data],
                name=f"{symbol} Price",
            ),
            row=1,
            col=1,
        )

        # Add trade markers
        if show_trades:
            buy_trades = [t for t in result["trades"] if t["order_type"] == "buy"]
            sell_trades = [t for t in result["trades"] if t["order_type"] == "sell"]

            if buy_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(t["timestamp"]) for t in buy_trades],
                        y=[t["price"] for t in buy_trades],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="green"),
                        name="Buy",
                        text=[f"Buy {t['shares']} shares" for t in buy_trades],
                        hovertemplate="%{text}<br>Price: $%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            if sell_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(t["timestamp"]) for t in sell_trades],
                        y=[t["price"] for t in sell_trades],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="red"),
                        name="Sell",
                        text=[f"Sell {t['shares']} shares" for t in sell_trades],
                        hovertemplate="%{text}<br>Price: $%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Add cumulative return curve
        portfolio_history = result["portfolio_history"]
        portfolio_dates = [pd.to_datetime(ph["date"]) for ph in portfolio_history]
        cumulative_returns = [ph["cumulative_return"] * 100 for ph in portfolio_history]

        fig.add_trace(
            go.Scatter(
                x=portfolio_dates,
                y=cumulative_returns,
                mode="lines",
                name="Strategy cumulative return",
                line=dict(color="blue"),
            ),
            row=2,
            col=1,
        )

        # Add benchmark line (Buy and hold)
        buy_hold_return = result["benchmark_comparison"]["buy_hold_return"]
        fig.add_hline(
            y=buy_hold_return * 100,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Buy and hold: {buy_hold_return * 100:.2f}%",
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f"Backtest results: {symbol} - {strategy_name}",
            template="plotly_white",
            height=600,
            xaxis_rangeslider_visible=False,
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig

    def get_backtest_summary(self, symbol: str, strategy_name: str) -> str:
        """
        Get backtest summary text

        Args:
            symbol: Trading symbol
            strategy_name: Strategy name

        Returns:
            Formatted backtest summary
        """
        result_key = f"{symbol}_{strategy_name}"

        if result_key not in self.current_results:
            return f"Cannot find {symbol} Strategy {strategy_name} backtest results"

        result = self.current_results[result_key]
        basic = result["basic_info"]
        metrics = result["performance_metrics"]
        benchmark = result["benchmark_comparison"]

        summary = f"""
Backtest summary report
{"=" * 60}

Basic information:
- Trading symbol: {basic["symbol"]}
- Strategy name: {basic["strategy_name"]}
- Backtest period: {basic["start_date"]} to {basic["end_date"]}
- Trading days: {basic["total_days"]}
- Initial capital: ${basic["initial_capital"]:,.0f}
- Maximum shares per buy: {basic["max_shares_per_trade"]}

Performance:
- Final assets: ${metrics["final_value"]:,.0f}
- Total return: {metrics["total_return"] * 100:.2f}%
- Annualized return: {metrics["annual_return"] * 100:.2f}%
- Annualized volatility: {metrics["volatility"] * 100:.2f}%
- Maximum drawdown: {metrics["max_drawdown"] * 100:.2f}%

Trading statistics:
- Number of trades: {metrics["num_trades"]}
- Win rate: {metrics["win_rate"] * 100:.2f}%

Benchmark comparison:
- Buy-and-hold return: {benchmark["buy_hold_return"] * 100:.2f}%
- Strategy excess return: {benchmark["alpha"] * 100:.2f}%
- Outperformed benchmark: {"Yes" if benchmark["outperformed"] else "No"}
        """.strip()

        return summary


# For backward compatibility, retain an alias for the original class name
BacktestEngine = CustomBacktestEngine
