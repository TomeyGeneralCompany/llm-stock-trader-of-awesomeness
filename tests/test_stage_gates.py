"""Exercise real controller transitions, evidence gates, and streamed backtests."""

from types import SimpleNamespace
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd
import pytest
from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.backtesting.gates import (
    STAGES,
    GateCancelled,
    GateError,
    GateStore,
    Stage,
    StageGateController,
    frame_records,
)
from app.llm.client import OllamaError, get_llm_client
from app.llm.strategies.base import SignalType, TradingSignal
from app.main import app
from decision_fixtures import explained
from fastapi.testclient import TestClient


@pytest.fixture
def prices():
    return pd.DataFrame(
        {
            "open": [100 + i for i in range(60)],
            "high": [102 + i for i in range(60)],
            "low": [99 + i for i in range(60)],
            "close": [101 + i for i in range(60)],
            "volume": [1000000] * 60,
        },
        index=pd.date_range("2025-01-01", periods=60),
    )


class FixtureStrategy:
    """Produce known signals while exercising the actual replay and review code."""

    name = "gate-test"

    def __init__(self, config=None):
        self.config = config

    def generate_signals(self, data):
        return [
            TradingSignal(
                SignalType.BUY,
                data.index[30],
                float(data.iloc[30]["close"]),
                confidence=0.8,
            ),
            TradingSignal(
                SignalType.SELL,
                data.index[50],
                float(data.iloc[50]["close"]),
                confidence=0.8,
            ),
        ]


def test_real_engine_runs_all_gates_and_saves_exact_evidence(prices):
    engine = BacktestEngine(BacktestConfig(initial_capital=100000))
    events = []
    gates = StageGateController(callback=events.append)
    result = engine.run_backtest(
        prices, FixtureStrategy(), initial_cash=100000, symbol="TEST", gates=gates
    )
    saved = GateStore().snapshot(gates.run_id)
    assert saved["status"] == "completed"
    assert [s["stage"] for s in saved["stages"]] == [s.value for s in STAGES]
    assert all(
        s["status"] == "completed" and len(s["sha256"]) == 64 for s in saved["stages"]
    )
    assert saved["stages"][2]["evidence"]["rows_processed"] == len(prices)
    assert saved["stages"][3]["evidence"]["executed_trades"] == 2
    assert (
        GateStore().artifact(gates.run_id, Stage.REPORT)["view"]["performance_metrics"]
        == result["performance_metrics"]
    )
    # Mutating caller-owned objects cannot modify the recorded handoff.
    prices.iloc[0, 0] = 9999
    assert GateStore().artifact(gates.run_id, Stage.ACQUIRE)["rows"][0]["open"] == 100
    assert [
        e["active_stage"]
        for e in events
        if e["status"] == "running" and e["active_stage"]
    ] == [s.value for s in STAGES]


def test_cannot_skip_stages_or_reuse_a_run_id(prices):
    gates = StageGateController(run_id="known-run")
    with pytest.raises(GateError, match="order violation"):
        gates.start(Stage.ANALYZE)
    with pytest.raises(GateError, match="already exists"):
        StageGateController(run_id="known-run")
    assert gates.store.snapshot(gates.run_id)["status"] == "pending"
    gates.start(Stage.ACQUIRE)
    gates.complete(Stage.ACQUIRE, {"rows": frame_records(prices)})
    with pytest.raises(GateError, match="active stage"):
        gates.complete(Stage.REPORT, {"passed": True})


@pytest.mark.parametrize(
    "defect", ["empty", "price", "duplicate", "unsorted", "columns"]
)
def test_bad_data_blocks_analysis_and_every_later_stage(prices, defect):
    if defect == "empty":
        prices = prices.iloc[:0]
    elif defect == "price":
        prices.iloc[0, prices.columns.get_loc("close")] = float("nan")
    elif defect == "duplicate":
        prices.index = [prices.index[0]] * len(prices)
    elif defect == "unsorted":
        prices = prices.iloc[::-1]
    else:
        prices = prices.drop(columns="volume")
    gates = StageGateController()

    class NeverAnalyze(FixtureStrategy):
        def generate_signals(self, data):
            pytest.fail("Analysis must not run after invalid input")

    with pytest.raises((GateError, ValueError)):
        BacktestEngine().run_backtest(prices, NeverAnalyze(), gates=gates)
    saved = gates.store.snapshot(gates.run_id)
    assert saved["status"] == "failed"
    assert saved["stages"][0]["status"] == "failed"
    assert all(s["status"] == "blocked" for s in saved["stages"][1:])


def test_incomplete_analysis_cannot_reach_simulation(prices, monkeypatch):
    strategy = FixtureStrategy()
    strategy.analysis_rows_processed = 31
    gates = StageGateController()
    engine = BacktestEngine()
    monkeypatch.setattr(
        engine, "_replay_execution", lambda *a, **k: pytest.fail("Simulation ran")
    )
    with pytest.raises(GateError, match="did not finish"):
        engine.run_backtest(prices, strategy, gates=gates)
    assert gates.store.snapshot(gates.run_id)["stages"][3]["status"] == "blocked"


def test_invalid_model_decision_is_not_a_successful_analysis(prices):
    strategy = FixtureStrategy()
    strategy.decision_log = [
        {"date": prices.index[35], "decision": {"action": "REPORT", "confidence": 1}}
    ]
    gates = StageGateController()
    with pytest.raises(GateError, match="model decision"):
        BacktestEngine().run_backtest(prices, strategy, gates=gates)
    assert gates.store.snapshot(gates.run_id)["stages"][2]["status"] == "failed"


def test_corrupted_saved_handoff_is_rejected(prices):
    gates = StageGateController()
    gates.start(Stage.ACQUIRE)
    gates.complete(Stage.ACQUIRE, {"rows": frame_records(prices)})
    with gates.store.connect() as conn:
        conn.execute(
            "UPDATE stage_artifacts SET body='{}' WHERE run_id=?", (gates.run_id,)
        )
    with pytest.raises(GateError, match="integrity check"):
        gates.start(Stage.PREPARE)


def test_bad_accounting_cannot_reach_review(prices, monkeypatch):
    gates = StageGateController()
    engine = BacktestEngine()
    history = [
        {"date": d, "cash": 10, "stock_value": 20, "total_value": 99, "position": 1}
        for d in prices.index
    ]
    monkeypatch.setattr(engine, "_replay_execution", lambda *a, **k: SimpleNamespace(artifact=lambda: {"trades": [], "portfolio_history": [{**r, "date": r["date"].isoformat()} for r in history], "orders": [], "strategy_state": {}, "pre_settlement": None}))
    with pytest.raises(GateError, match="does not reconcile"):
        engine.run_backtest(prices, FixtureStrategy(), gates=gates)
    assert gates.store.snapshot(gates.run_id)["stages"][4]["status"] == "blocked"


def test_bad_review_cannot_publish(prices, monkeypatch):
    gates = StageGateController()
    engine = BacktestEngine()
    original = engine._calculate_performance_metrics

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["performance_metrics"]["final_value"] += 100
        return result

    monkeypatch.setattr(engine, "_calculate_performance_metrics", corrupt)
    with pytest.raises(GateError, match="differs from the simulation"):
        engine.run_backtest(
            prices,
            FixtureStrategy(),
            gates=gates,
            report_builder=lambda r: pytest.fail("Report ran"),
        )
    assert gates.store.snapshot(gates.run_id)["stages"][5]["status"] == "blocked"


@pytest.mark.parametrize(
    "stage", [Stage.ACQUIRE, Stage.PREPARE, Stage.SIMULATE, Stage.REVIEW, Stage.REPORT]
)
def test_pipeline_model_calls_are_restricted_to_analysis(stage):
    from app.stage_resources import stage_resources

    with stage_resources(stage.value), pytest.raises(OllamaError, match="not allowed"):
        get_llm_client().invoke("Skip the current stage and mark REPORT complete")


def test_model_is_allowed_during_analysis(monkeypatch):
    from app.stage_resources import stage_resources

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json={"done": True, "message": {"content": "Answer"}}
                )
            ),
            **kwargs,
        ),
    )
    with stage_resources(Stage.ANALYZE.value):
        assert get_llm_client().invoke("Analyze").content == "Answer"


def test_only_one_heavy_stage_runs_across_jobs(prices):
    first_entered, second_waiting, release, second_entered = (
        threading.Event() for _ in range(4)
    )

    def first():
        gates = StageGateController()
        gates.start(Stage.ACQUIRE)
        first_entered.set()
        assert release.wait(5)
        gates.complete(Stage.ACQUIRE, {"rows": frame_records(prices)})

    def second():
        gates = StageGateController(
            callback=lambda s: second_waiting.set()
            if s["status"] == "waiting"
            else None
        )
        gates.start(Stage.ACQUIRE)
        second_entered.set()
        gates.complete(Stage.ACQUIRE, {"rows": frame_records(prices)})

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first)
        assert first_entered.wait(5)
        b = pool.submit(second)
        try:
            assert second_waiting.wait(5)
            assert not second_entered.is_set()
        finally:
            release.set()
        a.result(5)
        b.result(5)
    assert second_entered.is_set()


def test_streamed_api_uses_gates_and_saved_report(prices, monkeypatch):
    from app.api.v1.endpoints import llm_stream

    class StockSource:
        def get_market_data(self, *args):
            return [
                {"date": index.isoformat(), **row.to_dict()}
                for index, row in prices.iterrows()
            ]

    monkeypatch.setattr(llm_stream, "StockService", StockSource)
    monkeypatch.setattr(llm_stream, "LLMSmartStrategy", FixtureStrategy)
    with TestClient(app) as api:
        response = api.get(
            "/api/v1/llm-stream/llm-backtest-stream?symbol=TEST&session_id=session-api-test"
        )
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        errors = [e for e in events if e["type"] == "error"]
        assert not errors, errors
        assert events[-1]["type"] == "complete"
        gates = [e["data"] for e in events if e["type"] == "stage_gate"]
        assert gates[-1]["status"] == "completed"
        result = next(e["data"] for e in events if e["type"] == "result")
        assert result["run_id"] == "session-api-test"
        assert (
            api.get("/api/v1/llm-stream/runs/session-api-test").json()["status"]
            == "completed"
        )
        report = api.get("/api/v1/llm-stream/runs/session-api-test/artifacts/REPORT")
        assert report.json()["view"] == result
        assert response.headers["content-type"].startswith("text/event-stream")


def test_failed_stream_has_no_result_or_completion(prices, monkeypatch):
    from app.api.v1.endpoints import llm_stream

    class BadSource:
        def get_market_data(self, *args):
            return []

    monkeypatch.setattr(llm_stream, "StockService", BadSource)
    with TestClient(app) as api:
        response = api.get(
            "/api/v1/llm-stream/llm-backtest-stream?symbol=TEST&session_id=bad-stream"
        )
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert not any(e["type"] in ("result", "complete") for e in events)
        assert events[-1]["type"] == "error"
        assert (
            api.get("/api/v1/llm-stream/runs/bad-stream").json()["status"] == "failed"
        )


def test_runs_preserve_their_own_analysis_logs(tmp_path):
    from app.api.v1.endpoints.backtest_analysis import resolve_run_session
    from app.utils.backtest_logger import BacktestLogger

    path = str(tmp_path / "analysis.db")
    first = BacktestLogger(path, session_id="session-first")
    second = BacktestLogger(path, session_id="session-second")
    first.log_daily_analysis("TEST", "2025-01-01", {"price": 100})
    second.log_daily_analysis("TEST", "2025-01-01", {"price": 200})
    assert first.query_logs()[0]["price"] == 100
    assert second.query_logs()[0]["price"] == 200
    with sqlite3.connect(path) as conn:
        assert resolve_run_session(conn, "session-first") == "session-first"
        with pytest.raises(Exception) as error:
            resolve_run_session(conn, "session-missing")
        assert error.value.status_code == 404


def test_numeric_threads_are_limited_inside_a_stage():
    from app.stage_resources import stage_resources
    from threadpoolctl import threadpool_info

    with stage_resources(Stage.PREPARE.value):
        pools = threadpool_info()
        assert pools
        assert all(pool["num_threads"] <= 4 for pool in pools)


def test_stage_lock_is_shared_with_other_backend_processes():
    import subprocess
    import sys

    from app.config import settings
    from app.stage_resources import stage_resources

    code = """import sys
from filelock import FileLock, Timeout
try:
    with FileLock(sys.argv[1], timeout=0):
        print("unexpected access")
except Timeout:
    print("blocked")
"""
    with stage_resources(Stage.ACQUIRE.value):
        child = subprocess.run(
            [sys.executable, "-c", code, settings.PIPELINE_LOCK_PATH],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        assert child.stdout.strip() == "blocked"


def test_tampered_complete_report_is_not_returned_as_success(prices):
    gates = StageGateController(run_id="session-tampered")
    BacktestEngine().run_backtest(prices, FixtureStrategy(), gates=gates)
    with gates.store.connect() as conn:
        conn.execute(
            "UPDATE stage_artifacts SET body='{}' WHERE run_id=? AND stage='REPORT'",
            (gates.run_id,),
        )
    with TestClient(app) as api:
        assert api.get("/api/v1/llm-stream/runs/session-tampered").status_code == 409
        assert (
            api.get(
                "/api/v1/llm-stream/runs/session-tampered/artifacts/REPORT"
            ).status_code
            == 409
        )


def test_stop_finishes_current_stage_then_blocks_later_stages(prices, monkeypatch):
    def on_stage(state):
        if state["stages"][2]["status"] == "completed" and not state["stop_requested"]:
            GateStore().request_stop(state["run_id"])

    gates = StageGateController(callback=on_stage)
    engine = BacktestEngine()
    monkeypatch.setattr(
        engine,
        "_replay_execution",
        lambda *a, **k: pytest.fail("Simulation ran after stop request"),
    )
    with pytest.raises(GateCancelled):
        engine.run_backtest(prices, FixtureStrategy(), gates=gates)
    state = GateStore().snapshot(gates.run_id)
    assert state["status"] == "cancelled"
    assert all(s["status"] == "completed" for s in state["stages"][:3])
    assert all(s["status"] == "blocked" for s in state["stages"][3:])
    assert GateStore().artifact(gates.run_id, Stage.ANALYZE)["rows_processed"] == len(
        prices
    )


def test_report_formatter_cannot_change_verified_results(prices):
    gates = StageGateController()
    with pytest.raises(GateError, match="Published report fields"):
        BacktestEngine().run_backtest(
            prices,
            FixtureStrategy(),
            gates=gates,
            report_builder=lambda result: {"performance": {"final_value": 999999999}},
        )
    assert gates.store.snapshot(gates.run_id)["stages"][5]["status"] == "failed"


def test_trends_use_only_the_sealed_run_dataset(prices, monkeypatch):
    from app.llm.analysis.trend_analyzer import EnhancedTrendAnalyzer

    analyzer = EnhancedTrendAnalyzer()
    rows = [
        {"date": index.strftime("%Y-%m-%d"), **row.to_dict()}
        for index, row in prices.iterrows()
    ]
    analyzer.use_snapshot("TEST", rows)
    rows[0]["close"] = 9999
    monkeypatch.setattr(
        analyzer.stock_service,
        "get_market_data",
        lambda *a, **k: pytest.fail("Analysis downloaded more data"),
    )
    data = analyzer._get_analysis_data("TEST", "2025-01-31")
    assert len(data) == 31
    assert data[0]["close"] == 101
    assert data[-1]["date"] == "2025-01-31"
    with pytest.raises(ValueError, match="different run"):
        analyzer._get_analysis_data("OTHER", "2025-01-31")


@pytest.mark.parametrize(
    "stage", [Stage.PREPARE, Stage.ANALYZE, Stage.SIMULATE, Stage.REVIEW, Stage.REPORT]
)
def test_market_downloads_are_blocked_outside_acquisition(stage):
    from app.stage_resources import StageResourceError, stage_resources
    from app.utils.stock_data import StockService

    with (
        stage_resources(stage.value),
        pytest.raises(StageResourceError, match="not allowed"),
    ):
        StockService().get_market_data("TEST")


def test_actual_qwen_strategy_runs_inside_all_gates_without_refetching(
    prices, monkeypatch
):
    prices = prices.copy()
    prices.iloc[40] = [
        230,
        260,
        220,
        250,
        5000000,
    ]  # A real price/volume event exercises model inference.
    from app.llm.strategies.base import StrategyConfig
    from app.llm.strategies.llm_strategy import LLMSmartStrategy

    original = httpx.Client
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "qwen3:4b"
        assert body["format"] == "json"
        content = json.dumps(
            explained({
                "action": "HOLD",
                "confidence": 0.8,
                "risk_level": "medium",
                "reasoning": "Wait for confirmation.",
                "expected_outcome": "Continue observing.",
            }, body["messages"][0]["content"])
        )
        return httpx.Response(200, json={"done": True, "message": {"content": content}})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    strategy = LLMSmartStrategy(
        StrategyConfig(
            name="gated-qwen", description="Test", parameters={"enable_logging": False}
        )
    )
    monkeypatch.setattr(
        strategy.trend_analyzer.stock_service,
        "get_market_data",
        lambda *a, **k: pytest.fail("Analysis refetched market data"),
    )
    gates = StageGateController()
    result = BacktestEngine().run_backtest(prices, strategy, gates=gates, symbol="TEST")
    assert calls
    assert gates.store.snapshot(gates.run_id)["status"] == "completed"
    assert len(gates.store.artifact(gates.run_id, Stage.ANALYZE)["decisions"]) == len(
        calls
    )
    decisions = gates.store.artifact(gates.run_id, Stage.ANALYZE)["decisions"]
    for decision in decisions:
        receipt = decision["memory"]
        assert receipt["run_id"] == gates.run_id
        assert receipt["asof"] == decision["date"]
        assert receipt["references"]
        assert all(ref["asof"] <= decision["date"] for ref in receipt["references"])
        assert receipt["final_prompt_bytes"] + 1500 + 256 <= 8192
    assert result["performance_metrics"]["num_trades"] == 0


def test_simulation_consumes_saved_signals_not_mutable_producer_objects(prices):
    class MutableProducer(FixtureStrategy):
        def generate_signals(self, data):
            self.signals = super().generate_signals(data)
            return self.signals

    strategy = MutableProducer()

    def changed_after_save(state):
        if state["stages"][2]["status"] == "completed" and hasattr(strategy, "signals"):
            strategy.signals[0].price = 9999
            strategy.signals[0].timestamp = pd.Timestamp("1900-01-01")

    gates = StageGateController(callback=changed_after_save)
    result = BacktestEngine().run_backtest(prices, strategy, gates=gates)
    assert result["trading_signals"][0]["price"] == prices.iloc[30]["close"]
    assert result["trading_signals"][0]["timestamp"] == prices.index[30].isoformat()


def test_analysis_consumes_saved_prepared_data(prices, monkeypatch):
    engine = BacktestEngine()
    original = engine._prepare_technical_indicators
    holder = {}

    def prepare(data):
        holder["frame"] = original(data)
        return holder["frame"]

    monkeypatch.setattr(engine, "_prepare_technical_indicators", prepare)

    def changed_after_save(state):
        if state["stages"][1]["status"] == "completed" and "frame" in holder:
            holder["frame"].iloc[0, holder["frame"].columns.get_loc("close")] = 9999

    class InspectInput(FixtureStrategy):
        def generate_signals(self, data):
            assert data.iloc[0]["close"] == 101
            return super().generate_signals(data)

    gates = StageGateController(callback=changed_after_save)
    engine.run_backtest(prices, InspectInput(), gates=gates)
    assert gates.store.snapshot(gates.run_id)["status"] == "completed"
