"""
Backtesting Engine Core Module

Provides complete strategy backtesting functionality, including portfolio management,
trade execution simulation, and performance metrics calculation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..llm.strategies import SignalType, TradingSignal, TradingStrategy


class OrderType(Enum):
    """Order Types"""

    MARKET = "market"  # Market order
    LIMIT = "limit"  # Limit order
    STOP = "stop"  # Stop order
    STOP_LIMIT = "stop_limit"  # Stop limit order


class OrderStatus(Enum):
    """Order Status"""

    PENDING = "pending"  # Pending
    FILLED = "filled"  # Filled
    CANCELLED = "cancelled"  # Cancelled
    REJECTED = "rejected"  # Rejected


@dataclass
class Order:
    """
    Order data class

    Attributes:
        order_id: Order ID
        symbol: Trading symbol
        order_type: Order type
        side: Trade direction (BUY/SELL)
        quantity: Trade quantity
        price: Order price (For market orders use None)
        timestamp: Order submission time
        status: Order status
        filled_price: Fill price
        filled_quantity: Filled quantity
        commission: Commission
        metadata: Additional information
    """

    order_id: str
    symbol: str
    order_type: OrderType
    side: SignalType
    quantity: float
    price: Optional[float]
    timestamp: pd.Timestamp
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """
    Position data class

    Attributes:
        symbol: Trading symbol
        quantity: Position quantity (Positive for long positions, Negative for short positions)
        avg_cost: Average cost
        market_value: Market value
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L"""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity


@dataclass
class BacktestConfig:
    """
    Backtest configuration class

    Attributes:
        initial_capital: Initial capital
        commission_rate: Commission rate (0.001 = 0.1%)
        slippage_rate: Slippage rate (0.0005 = 0.05%)
        min_trade_amount: Minimum trade value
        max_position_size: Maximum position allocation (0.5 = 50%)
        enable_short_selling: Whether short selling is allowed
        margin_requirement: Margin requirement (Used for short selling)
        risk_free_rate: Risk-free interest rate (Annualized)
    """

    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_trade_amount: float = 100.0
    max_position_size: float = 1.0
    enable_short_selling: bool = False
    margin_requirement: float = 0.5
    risk_free_rate: float = 0.02


class Portfolio:
    """
    Portfolio manager

    Manage cash, Positions, and order history in the portfolio state
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize portfolio

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders_history: List[Order] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[Tuple[pd.Timestamp, float]] = []

        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = config.initial_capital

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current total portfolio value

        Args:
            current_prices: Dictionary of current prices {symbol: price}

        Returns:
            Total portfolio value
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_market_value(current_prices[symbol])
                total_value += position.market_value

        return total_value

    def get_available_cash(self) -> float:
        """Get available cash"""
        return self.cash

    def get_position(self, symbol: str) -> Position:
        """
        Get position information

        Args:
            symbol: Trading symbol

        Returns:
            Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def calculate_position_size(
        self, symbol: str, signal: TradingSignal, current_price: float
    ) -> float:
        """
        Calculate trade quantity

        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current price

        Returns:
            Suggested trade quantity
        """
        portfolio_value = self.get_portfolio_value({symbol: current_price})

        # Calculate position size from confidence and maximum position allocation
        confidence_factor = signal.confidence
        max_position_value = (
            portfolio_value * self.config.max_position_size * confidence_factor
        )

        if signal.signal_type == SignalType.BUY:
            # Buy signal
            available_cash = self.get_available_cash()
            trade_value = min(max_position_value, available_cash * 0.95)  # Reserve 5%Cash

            if trade_value < self.config.min_trade_amount:
                return 0.0

            # Account for commission and slippage
            effective_price = current_price * (
                1 + self.config.commission_rate + self.config.slippage_rate
            )
            quantity = trade_value / effective_price

        elif signal.signal_type == SignalType.SELL:
            # Sell signal
            current_position = self.get_position(symbol)
            if current_position.quantity <= 0:
                return 0.0

            # Determine sale proportion based on confidence
            sell_ratio = confidence_factor
            quantity = current_position.quantity * sell_ratio

        else:
            return 0.0

        return max(0, quantity)

    def place_order(
        self,
        symbol: str,
        side: SignalType,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Order:
        """
        Place order

        Args:
            symbol: Trading symbol
            side: Trade direction
            quantity: Trade quantity
            order_type: Order type
            price: Limit order price
            timestamp: Order submission time

        Returns:
            Order object
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        order_id = f"{symbol}_{side.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )

        self.orders_history.append(order)
        return order

    def execute_order(self, order: Order, market_price: float) -> bool:
        """
        Execute order

        Args:
            order: Order object
            market_price: Market price

        Returns:
            Whether execution succeeded
        """
        if order.status != OrderStatus.PENDING:
            return False

        # Calculate actual fill price (Account for slippage)
        if order.side == SignalType.BUY:
            slippage = market_price * self.config.slippage_rate
            filled_price = market_price + slippage
        else:  # SELL
            slippage = market_price * self.config.slippage_rate
            filled_price = market_price - slippage

        # Calculate commission
        trade_value = order.quantity * filled_price
        commission = trade_value * self.config.commission_rate

        # Check available funds
        if order.side == SignalType.BUY:
            required_cash = trade_value + commission
            if required_cash > self.cash:
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reason"] = "Insufficient funds"
                return False

        # Execute trade
        position = self.get_position(order.symbol)

        if order.side == SignalType.BUY:
            # Buy
            total_cost = position.quantity * position.avg_cost
            new_total_cost = total_cost + trade_value
            new_quantity = position.quantity + order.quantity

            if new_quantity > 0:
                position.avg_cost = new_total_cost / new_quantity
            position.quantity = new_quantity

            self.cash -= trade_value + commission

        else:  # SELL
            # Sell
            if order.quantity > position.quantity:
                # Cannot sell more shares than held
                order.quantity = position.quantity
                trade_value = order.quantity * filled_price
                commission = trade_value * self.config.commission_rate

            # Calculate realized P&L
            realized_pnl = (filled_price - position.avg_cost) * order.quantity
            position.realized_pnl += realized_pnl
            position.quantity -= order.quantity

            self.cash += trade_value - commission

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_price = filled_price
        order.filled_quantity = order.quantity
        order.commission = commission

        # Update statistics
        self.total_commission_paid += commission
        self.total_slippage_cost += abs(filled_price - market_price) * order.quantity

        return True

    def update_portfolio_value(
        self, timestamp: pd.Timestamp, current_prices: Dict[str, float]
    ) -> None:
        """
        Update portfolio value history

        Args:
            timestamp: Timestamp
            current_prices: Current price
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        self.portfolio_values.append((timestamp, portfolio_value))

        # Update maximum drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        else:
            current_drawdown = (
                self.peak_portfolio_value - portfolio_value
            ) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Calculate daily returns
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2][1]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary

        Returns:
            Portfolio summary dictionary
        """
        if not self.portfolio_values:
            return {}

        current_value = self.portfolio_values[-1][1]
        total_return = (current_value - self.initial_capital) / self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_value": current_value,
            "cash": self.cash,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "total_commission": self.total_commission_paid,
            "total_slippage": self.total_slippage_cost,
            "num_trades": len(
                [o for o in self.orders_history if o.status == OrderStatus.FILLED]
            ),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
                if pos.quantity != 0
            },
        }
