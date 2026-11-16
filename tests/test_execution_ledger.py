"""Hand-calculated execution/accounting regressions; no broker or model required."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.backtesting.ledger import ExecutionLedger, ExecutionSettings, replay
from app.llm.strategies.base import SignalType, TradingSignal, StrategyConfig
from app.llm.strategies.llm_strategy import LLMSmartStrategy


def prices(values):
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 100000} for o, h, l, c in values],
        index=pd.date_range("2025-01-01", periods=len(values)),
    )


def signal(data, day, action, quantity=None):
    return TradingSignal(SignalType[action], data.index[day], float(data.close.iloc[day]), quantity=quantity, confidence=.9)


def settings(**kwargs):
    return ExecutionSettings(**{"initial_cash": 1000, "commission_rate": .01,
        "max_position_size": 1., "stop_loss": 0., "take_profit": 0., **kwargs})


def test_next_open_gaps_affordability_fees_and_cancelled_last_decision():
    data = prices([(10, 11, 9, 10), (20, 22, 19, 21), (30, 31, 29, 30)])
    ledger = replay(data, [signal(data, 0, "BUY"), signal(data, 1, "SELL"), signal(data, 2, "BUY")], settings(), "T")
    buy, sell = ledger.trades
    assert (buy["timestamp"], buy["price"], buy["shares"]) == (data.index[1].isoformat(), 20, 49)
    assert buy["commission"] == pytest.approx(9.8)
    assert sell["price"] == 30 and sell["shares"] == 49
    assert sell["commission"] == pytest.approx(14.7)
    assert sell["realized_pnl"] == pytest.approx(465.5)
    assert ledger.cash == pytest.approx(1465.5)
    assert ledger.state()["total_trades"] == 1
    assert ledger.orders[-1]["status"] == "cancelled"
    assert all(t["decision_timestamp"] < t["timestamp"] for t in ledger.trades)


def test_allocation_is_enforced_at_fill_after_fees():
    data = prices([(10, 11, 9, 10), (20, 22, 19, 21)])
    ledger = replay(data, [signal(data, 0, "BUY")], settings(max_position_size=.25, liquidate_at_end=False), "T")
    assert ledger.shares == 12
    at_fill_equity = ledger.cash + ledger.shares * 20
    assert ledger.shares * 20 <= .25 * at_fill_equity
    assert ledger.cash >= 0


@pytest.mark.parametrize("bar,reason,price", [
    ((8, 12, 7, 9), "stop_loss_gap", 8),
    ((10, 12, 8, 10), "stop_loss", 9),  # both touched: adverse-first convention
    ((10, 12, 9.5, 11), "take_profit", 11),
])
def test_protective_brackets_and_gap_fills(bar, reason, price):
    data = prices([(10, 10.5, 9.5, 10), (10, 10.5, 9.5, 10), bar])
    ledger = replay(data, [signal(data, 0, "BUY", 10)], settings(stop_loss=.1, take_profit=.1), "T")
    assert ledger.trades[-1]["reason"] == reason
    assert ledger.trades[-1]["price"] == pytest.approx(price)


def test_final_settlement_charges_fee_and_is_idempotent():
    data = prices([(10, 10, 10, 10), (10, 10, 10, 10)])
    ledger = replay(data, [signal(data, 0, "BUY", 10)], settings(), "T")
    assert [t["commission"] for t in ledger.trades] == [1, 1]
    assert ledger.cash == 998
    assert ledger.trades[-1]["execution_phase"] == "close"
    ledger.finish(data)
    assert len(ledger.trades) == 2 and ledger.cash == 998


def test_partial_exits_allocate_entry_fees_and_use_net_profit():
    data = prices([(10, 10, 10, 10)] * 4)
    ledger = replay(data, [signal(data, 0, "BUY", 10), signal(data, 1, "SELL", 4),
                           signal(data, 2, "SELL", 6)], settings(), "T")
    exits = ledger.trades[1:]
    assert [t["cost_basis"] for t in exits] == pytest.approx([40.4, 60.6])
    assert [t["realized_pnl"] for t in exits] == pytest.approx([-.8, -1.2])
    assert ledger.state()["winning_trades"] == 0
    assert ledger.cash == 998


def test_minimum_commission_and_no_arbitrary_share_cap():
    ledger = ExecutionLedger(settings(initial_cash=100000, commission_rate=0, min_commission=2), "T")
    assert ledger.quantity(1) == 99998
    limited = ExecutionLedger(replace(ledger.settings, max_shares_per_trade=7), "T")
    assert limited.quantity(1) == 7


@pytest.mark.parametrize("kwargs", [
    {"initial_cash": 0}, {"initial_cash": float("inf")}, {"commission_rate": -1},
    {"max_position_size": 0}, {"max_position_size": 1.1}, {"stop_loss": 1},
    {"take_profit": -1}, {"min_commission": float("nan")}, {"max_shares_per_trade": 0},
])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)


def test_performance_uses_run_capital_equity_returns_and_net_exits():
    data = prices([(10, 10, 10, 10), (10, 11, 10, 11), (11, 11, 11, 11)])
    class Static:
        name = "known"
        def generate_signals(self, frame):
            return [signal(frame, 0, "BUY", 10), signal(frame, 1, "SELL", 10)]
    result = BacktestEngine(BacktestConfig(initial_capital=99999)).run_backtest(
        data, Static(), initial_cash=1000, transaction_cost=.01, max_position_size=1,
        stop_loss=0, take_profit=0)
    assert result["performance_metrics"]["final_value"] == pytest.approx(1007.9)
    assert result["performance_metrics"]["total_return"] == pytest.approx(.0079)
    expected_returns = np.array([0, .009, 1007.9 / 1009 - 1])
    assert result["performance_metrics"]["volatility"] == pytest.approx(expected_returns.std(ddof=1) * np.sqrt(252))
    assert result["strategy_statistics"]["total_realized_pnl"] == pytest.approx(7.9)
    assert result["strategy_statistics"]["winning_trades"] == 1
    assert result["performance_metrics"]["annual_return"] == pytest.approx((1007.9 / 1000)**(365.25/2)-1)


def test_calibration_and_decision_prefix_are_independent_of_future(monkeypatch):
    captured = []
    original = LLMSmartStrategy._analyze_stock_characteristics
    def calibration(self, data):
        captured.append(data.copy())
        return original(self, data)
    monkeypatch.setattr(LLMSmartStrategy, "_analyze_stock_characteristics", calibration)
    data = prices([(100+i, 101+i, 99+i, 100+i) for i in range(36)])
    altered = data.copy()
    altered.iloc[31:] *= 2
    for frame in (data, altered):
        strategy = LLMSmartStrategy(StrategyConfig(name="warmup", description="test", parameters={"enable_logging": False}))
        strategy.set_current_symbol("T")
        strategy.trend_analyzer.use_snapshot("T", [{"date": d.strftime("%Y-%m-%d"), **r.to_dict()} for d, r in frame.iterrows()])
        monkeypatch.setattr(strategy, "_detect_trigger_events", lambda *a: [])
        strategy.generate_signals(frame)
    assert len(captured[0]) == len(captured[1]) == 30
    pd.testing.assert_frame_equal(captured[0], captured[1])


def test_current_recommendation_does_not_mutate_positions():
    strategy = LLMSmartStrategy(StrategyConfig(name="pure", description="test", parameters={"enable_logging": False, "initial_capital": 1000}))
    before = (strategy.cash, strategy.shares, strategy.current_position)
    decision = {"action": "BUY", "confidence": .9, "reasoning": "Enter now.",
                "explanation": {"execution": {"mode": "now"}}}
    result = strategy._create_signal_from_decision(decision, pd.Timestamp("2025-01-01"), 10)
    assert result is not None and 0 < result.quantity < 100
    assert (strategy.cash, strategy.shares, strategy.current_position) == before
