"""Deterministic finite-cash ledger shared by analysis and verification replay."""

import math
from datetime import datetime
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionSettings:
    initial_cash: float = 10000.0
    commission_rate: float = 0.001
    min_commission: float = 0.0
    max_position_size: float = 0.3
    stop_loss: float = 0.05
    take_profit: float = 0.1
    max_shares_per_trade: int | None = None
    liquidate_at_end: bool = True

    def __post_init__(self):
        for key in ("initial_cash", "commission_rate", "min_commission", "max_position_size", "stop_loss", "take_profit"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key + " must be finite.")
        if self.initial_cash <= 0 or not 0 <= self.commission_rate < 1 or self.min_commission < 0:
            raise ValueError("Capital must be positive; commissions must be nonnegative with rate below 1.")
        if not 0 < self.max_position_size <= 1 or not 0 <= self.stop_loss < 1 or self.take_profit < 0:
            raise ValueError("Allocation must be in (0,1], stop loss in [0,1), and take profit nonnegative.")
        cap = self.max_shares_per_trade
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError("An optional share cap must be a positive integer.")
        if not isinstance(self.liquidate_at_end, bool):
            raise ValueError("liquidate_at_end must be boolean.")

    def to_dict(self):
        return asdict(self)


class ExecutionLedger:
    """Own all fills and portfolio mutations.

    Close decisions fill at the following open. Protective orders can fill during
    that next bar: gaps use its open, otherwise the stop wins if both protective
    levels are touched. Final liquidation is an explicit, commission-bearing fill.
    """

    def __init__(self, settings, symbol):
        self.settings, self.symbol = settings, symbol
        self.cash, self.shares = settings.initial_cash, 0
        self.cost_basis, self.entry_price, self.entry_date = 0.0, 0.0, None
        self.pending, self.orders, self.trades, self.history = [], [], [], []
        self.cursor, self.finished = -1, False
        self.pre_settlement = None

    def fee(self, value):
        return max(self.settings.min_commission, value * self.settings.commission_rate)

    def quantity(self, price):
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Execution price must be finite and positive.")
        a, rate = self.settings.max_position_size, self.settings.commission_rate
        equity = self.cash + self.shares * price
        room = max(0.0, a * equity - self.shares * price)
        # Enforce the allocation limit after paying the entry commission as well.
        quantity = max(0, min(
            int(room / (price * (1 + a * rate))),
            int(max(0.0, room - a * self.settings.min_commission) / price),
            int(self.cash / (price * (1 + rate))),
            int(max(0.0, self.cash - self.settings.min_commission) / price),
        ))
        if self.settings.max_shares_per_trade is not None:
            quantity = min(quantity, self.settings.max_shares_per_trade)
        return quantity

    def submit(self, signal):
        if self.finished or not self.history or signal.timestamp.isoformat() != self.history[-1]["date"]:
            raise ValueError("An order must originate at the current completed bar.")
        self.pending.append(signal.to_dict())

    def _fill(self, action, quantity, price, date, reason, decision=None, phase="open", confidence=1.0):
        if quantity <= 0:
            return False
        value, allocated, pnl = quantity * price, 0.0, None
        fee = self.fee(value)
        if action == "buy":
            if value + fee > self.cash + 1e-9:
                raise ValueError("Execution would overdraw cash.")
            self.entry_price = (self.shares * self.entry_price + value) / (self.shares + quantity)
            self.entry_date = self.entry_date or date
            self.cost_basis += value + fee
            self.cash -= value + fee
            self.shares += quantity
            total_cost = value + fee
        else:
            if quantity > self.shares or self.cash + value < fee:
                return False
            allocated = self.cost_basis * quantity / self.shares
            pnl = value - fee - allocated
            self.cash += value - fee
            self.shares -= quantity
            self.cost_basis -= allocated
            total_cost = value - fee
            if not self.shares:
                self.entry_price, self.cost_basis, self.entry_date = 0.0, 0.0, None
        self.trades.append({
            "trade_id": f"T{len(self.trades)+1:06d}", "timestamp": date,
            "symbol": self.symbol, "order_type": action, "shares": quantity,
            "price": price, "commission": fee, "total_cost": total_cost,
            "status": "executed", "signal_confidence": confidence, "reason": reason,
            "decision_timestamp": decision, "execution_phase": phase,
            "realized_pnl": pnl, "cost_basis": allocated,
        })
        return True

    def advance(self, data, index):
        if self.finished or index < self.cursor or index >= len(data):
            raise ValueError("Ledger cannot move backwards, past its data, or beyond settlement.")
        for i in range(self.cursor + 1, index + 1):
            date, row = data.index[i].isoformat(), data.iloc[i]
            opening, high, low, close = (float(row[k]) for k in ("open", "high", "low", "close"))
            queued, self.pending = self.pending, []
            for order in queued:
                if order["timestamp"] >= date:
                    raise ValueError("A close decision cannot fill at the same or an earlier bar.")
                action = order["signal_type"].lower()
                quantity = self.quantity(opening) if action == "buy" else self.shares
                if order.get("quantity") is not None and order["quantity"] > 0:
                    quantity = min(quantity, int(order["quantity"]))
                filled = self._fill(action, quantity, opening, date, order.get("reason", ""),
                                    order["timestamp"], confidence=order.get("confidence", 0)) if action in ("buy", "sell") and float(row["volume"]) > 0 else False
                self.orders.append({"decision_timestamp": order["timestamp"],
                    "execution_timestamp": date if filled else None, "action": action,
                    "status": "filled" if filled else "rejected",
                    "reason": "next_open" if filled else "No affordable allocation or eligible holdings."})
            if self.shares and float(row["volume"]) > 0:
                stop = self.entry_price * (1 - self.settings.stop_loss) if self.settings.stop_loss else None
                target = self.entry_price * (1 + self.settings.take_profit) if self.settings.take_profit else None
                price, reason = None, None
                if stop is not None and opening <= stop:
                    price, reason = opening, "stop_loss_gap"
                elif target is not None and opening >= target:
                    price, reason = opening, "take_profit_gap"
                elif stop is not None and low <= stop:
                    price, reason = stop, "stop_loss"
                elif target is not None and high >= target:
                    price, reason = target, "take_profit"
                if price is not None:
                    self._fill("sell", self.shares, price, date, reason,
                               phase="open" if reason.endswith("gap") else "protective")
            self.cursor = i
            value = self.cash + self.shares * close
            self.history.append({"date": date, "cash": self.cash, "position": self.shares,
                "stock_value": self.shares * close, "total_value": value,
                "cumulative_return": value / self.settings.initial_cash - 1})

    def finish(self, data):
        if self.finished:
            return
        self.advance(data, len(data) - 1)
        self.pre_settlement = dict(self.history[-1])
        for order in self.pending:
            self.orders.append({"decision_timestamp": order["timestamp"],
                "execution_timestamp": None, "action": order["signal_type"].lower(),
                "status": "cancelled", "reason": "No next bar is available."})
        self.pending = []
        if self.shares and self.settings.liquidate_at_end and float(data.iloc[-1]["volume"]) > 0:
            date, close = data.index[-1].isoformat(), float(data.iloc[-1]["close"])
            self._fill("sell", self.shares, close, date, "final_liquidation", phase="close")
            value = self.cash + self.shares * close
            self.history[-1].update(cash=self.cash, position=self.shares,
                stock_value=self.shares * close, total_value=value,
                cumulative_return=value / self.settings.initial_cash - 1)
        self.finished = True

    def state(self):
        exits = [t for t in self.trades if t["order_type"] == "sell"]
        return {"cash": self.cash, "shares": self.shares,
            "current_position": "long" if self.shares else None,
            "position_entry_price": self.entry_price, "position_entry_date": self.entry_date,
            "initial_capital": self.settings.initial_cash, "total_trades": len(exits),
            "winning_trades": sum(t["realized_pnl"] > 0 for t in exits),
            "total_realized_pnl": sum(t["realized_pnl"] for t in exits),
            "trade_returns": [100 * t["realized_pnl"] / t["cost_basis"] for t in exits if t["cost_basis"] > 0]}

    def sync(self, strategy):
        for key, value in self.state().items():
            if key == "position_entry_date" and value is not None:
                value = datetime.fromisoformat(value)
            setattr(strategy, key, value.copy() if isinstance(value, list) else value)
        if self.history:
            strategy.current_portfolio_value = self.history[-1]["total_value"]
            strategy.max_portfolio_value = max(self.settings.initial_cash, *(r["total_value"] for r in self.history))

    def artifact(self):
        return {"trades": self.trades, "portfolio_history": self.history,
                "orders": self.orders, "strategy_state": self.state(),
                "pre_settlement": self.pre_settlement}


def replay(data, signals, settings, symbol):
    ledger, by_date = ExecutionLedger(settings, symbol), {}
    for signal in signals:
        by_date.setdefault(signal.timestamp.isoformat(), []).append(signal)
    if set(by_date) - {date.isoformat() for date in data.index}:
        raise ValueError("A signal lies outside the acquired data.")
    for i, date in enumerate(data.index):
        ledger.advance(data, i)
        for signal in by_date.get(date.isoformat(), []):
            ledger.submit(signal)
    ledger.finish(data)
    return ledger



def validate_accounting(output, analysis):
    """Independent arithmetic/timing check, not a second invocation of the fill kernel."""
    settings = ExecutionSettings(**analysis["execution_settings"])
    if output["execution_settings"] != analysis["execution_settings"]:
        raise ValueError("Execution settings changed after analysis.")
    cash, shares, basis, entry_price = settings.initial_cash, 0, 0., 0.
    by_date = {}
    dates, prices = analysis["timestamps"], analysis["execution_prices"]
    def equal(actual, expected, label):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("Ledger does not reconcile: "+label)
    for trade in output["trades"]:
        by_date.setdefault(trade["timestamp"], []).append(trade)
    for index, record in enumerate(output["portfolio_history"]):
        row = prices[index]
        for trade in by_date.get(record["date"], []):
            quantity, price = trade["shares"], trade["price"]
            if type(quantity) is not int or quantity <= 0:
                raise ValueError("Executed shares must be a positive integer.")
            if row["volume"] <= 0:
                raise ValueError("Cannot execute against a zero-volume bar.")
            fee = max(settings.min_commission, quantity*price*settings.commission_rate)
            equal(trade["commission"], fee, "commission")
            decision = trade.get("decision_timestamp")
            if decision is not None:
                if index == 0 or decision != dates[index-1] or trade["execution_phase"] != "open":
                    raise ValueError("A decision must execute at the next observed open.")
                equal(price, row["open"], "opening execution price")
            elif trade["reason"] == "final_liquidation":
                if index != len(dates)-1 or trade["execution_phase"] != "close":
                    raise ValueError("Final settlement must occur at the last close.")
                equal(price, row["close"], "settlement price")
            elif trade["reason"] not in ("stop_loss", "stop_loss_gap", "take_profit", "take_profit_gap"):
                raise ValueError("Execution lacks an authorized decision or protective trigger.")
            if decision is None and trade["reason"] != "final_liquidation":
                stop = entry_price*(1-settings.stop_loss) if settings.stop_loss else None
                target = entry_price*(1+settings.take_profit) if settings.take_profit else None
                reason, trigger = None, None
                if shares and stop is not None and row["open"] <= stop:
                    reason, trigger = "stop_loss_gap", row["open"]
                elif shares and target is not None and row["open"] >= target:
                    reason, trigger = "take_profit_gap", row["open"]
                elif shares and stop is not None and row["low"] <= stop:
                    reason, trigger = "stop_loss", stop
                elif shares and target is not None and row["high"] >= target:
                    reason, trigger = "take_profit", target
                if trade["order_type"] != "sell" or trade["reason"] != reason or quantity != shares:
                    raise ValueError("Protective execution lacks the configured price trigger.")
                equal(price, trigger, "protective execution price")
            if trade["order_type"] == "buy":
                if settings.max_shares_per_trade is not None and quantity > settings.max_shares_per_trade:
                    raise ValueError("Executed buy exceeds the configured share cap.")
                entry_price = (shares*entry_price + quantity*price)/(shares+quantity)
                cash -= quantity*price + fee
                shares += quantity
                basis += quantity*price + fee
                equal(trade["total_cost"], quantity*price+fee, "buy cost")
                if shares*price > settings.max_position_size*(cash+shares*price)+1e-7:
                    raise ValueError("Entry exceeds the configured allocation cap.")
            else:
                if quantity > shares:
                    raise ValueError("Sell exceeds owned shares.")
                allocated = basis*quantity/shares
                pnl = quantity*price-fee-allocated
                equal(trade["cost_basis"], allocated, "allocated entry cost")
                equal(trade["realized_pnl"], pnl, "net realized profit")
                equal(trade["total_cost"], quantity*price-fee, "net sale proceeds")
                cash += quantity*price-fee
                shares -= quantity
                basis -= allocated
                if not shares:
                    entry_price = 0.
            if cash < -1e-7:
                raise ValueError("Execution overdrew cash.")
        equal(record["cash"], cash, "daily cash")
        equal(record["position"], shares, "daily holdings")
        equal(record["stock_value"], shares*row["close"], "marked stock value")
        equal(record["total_value"], cash+shares*row["close"], "equity")
        equal(record["cumulative_return"], (cash+shares*row["close"])/settings.initial_cash-1, "return")
    equal(output["strategy_state"]["cash"], cash, "final state cash")
    equal(output["strategy_state"]["shares"], shares, "final state holdings")
