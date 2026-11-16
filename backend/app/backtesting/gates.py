"""Controller-owned stage gates and durable, validated handoff artifacts."""

import hashlib
import json
import logging
import math
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from app.config import settings
from app.stage_resources import stage_resources


class Stage(str, Enum):
    ACQUIRE = "ACQUIRE"
    PREPARE = "PREPARE"
    ANALYZE = "ANALYZE"
    SIMULATE = "SIMULATE"
    REVIEW = "REVIEW"
    REPORT = "REPORT"


STAGES = tuple(Stage)


class GateError(RuntimeError):
    """A stage prerequisite, output, or persisted artifact failed validation."""


class GateCancelled(RuntimeError):
    """The owner requested a stop at the next stage boundary."""


def json_value(value: Any) -> Any:
    """Encode stage evidence without silently stringifying unknown objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        # Indicator warm-up values may be NaN; gate validators still require
        # finite prices, balances, and published performance metrics.
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    raise GateError(f"Unsupported stage evidence type: {type(value).__name__}")


def frame_records(frame: pd.DataFrame) -> list[dict]:
    return json_value(
        [dict(timestamp=index, **row.to_dict()) for index, row in frame.iterrows()]
    )


def records_frame(records: list[dict]) -> pd.DataFrame:
    """Rebuild a consumer's input from a verified handoff, not producer memory."""
    frame = pd.DataFrame(records)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp")))
    return frame


def _canonical(value: Any) -> str:
    return json.dumps(
        json_value(value), sort_keys=True, allow_nan=False, separators=(",", ":")
    )


def _finite(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise GateError(f"{name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise GateError(f"{name} must be at least {minimum}.")
    return value


def _rows(rows: Any) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise GateError("No market data was produced.")
    dates = []
    for row in rows:
        missing = {"timestamp", "open", "high", "low", "close", "volume"} - row.keys()
        if missing:
            raise GateError(
                f"Market data is missing fields: {', '.join(sorted(missing))}."
            )
        try:
            dates.append(pd.Timestamp(row["timestamp"]))
        except (TypeError, ValueError) as exc:
            raise GateError("Market data contains an invalid timestamp.") from exc
        if pd.isna(dates[-1]):
            raise GateError("Market data contains an invalid timestamp.")
        prices = {
            k: _finite(row[k], k, 0.000000001) for k in ("open", "high", "low", "close")
        }
        _finite(row["volume"], "volume", 0)
        if prices["high"] < max(prices.values()) or prices["low"] > min(
            prices.values()
        ):
            raise GateError("Market data has inconsistent OHLC prices.")
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise GateError(
            "Market data must have unique timestamps in chronological order."
        )
    return rows


def validate_output(stage: Stage, output: dict, previous: Optional[dict]) -> dict:
    """Validate actual output; model claims never authorize a transition."""
    if not isinstance(output, dict):
        raise GateError("Stage output must be an object.")
    if stage in (Stage.ACQUIRE, Stage.PREPARE):
        rows = _rows(output["rows"])
        if stage == Stage.PREPARE:
            source = previous["rows"]
            if [r["timestamp"] for r in rows] != [r["timestamp"] for r in source]:
                raise GateError("Preparation changed market-data coverage.")
            for prepared, original in zip(rows, source):
                if any(
                    prepared[k] != original[k]
                    for k in ("open", "high", "low", "close", "volume")
                ):
                    raise GateError(
                        "Preparation changed source market prices or volume."
                    )
            if not all(k in rows[-1] for k in ("rsi", "macd", "bb_upper")):
                raise GateError("Required technical indicators are missing.")
        return {
            "observations": len(rows),
            "first": rows[0]["timestamp"],
            "last": rows[-1]["timestamp"],
        }
    if stage == Stage.ANALYZE:
        source = previous["rows"]
        if output["rows_processed"] != len(source):
            raise GateError("Analysis did not finish processing the prepared data.")
        if output["timestamps"] != [r["timestamp"] for r in source]:
            raise GateError("Analysis changed the prepared data period.")
        dates = {r["timestamp"] for r in source}
        for signal in output["signals"]:
            if signal["timestamp"] not in dates or signal["signal_type"] not in (
                "BUY",
                "SELL",
                "HOLD",
            ):
                raise GateError(
                    "Analysis produced an unsupported or out-of-range signal."
                )
            _finite(signal["price"], "signal price", 0.000000001)
            confidence = _finite(signal["confidence"], "signal confidence", 0)
            if confidence > 1:
                raise GateError("Signal confidence exceeds 1.")
        for log in output["decisions"]:
            if log.get("date") not in dates:
                raise GateError("A model decision belongs to a different data period.")
            decision = log.get("decision")
            from app.llm.decisions import DecisionError, validate_decision

            try:
                context = log.get("context")
                if log.get("memory"):
                    from app.llm.memory import digest

                    if not isinstance(context, dict) or context.get("asof") != log["date"]:
                        raise DecisionError("A model decision is missing its dated controller context.")
                    body = json.dumps(context, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    if digest(body) != log["memory"]["working_sha256"]:
                        raise DecisionError("A model decision context does not match its memory receipt.")
                validate_decision(decision, context=context)
            except DecisionError as exc:
                raise GateError(str(exc)) from exc
        if "execution_settings" in output:
            expected = [{key: row[key] for key in ("open", "high", "low", "close", "volume")} for row in source]
            if output.get("execution_prices") != expected:
                raise GateError("Execution prices changed from prepared evidence.")
            ledger = output.get("decision_ledger")
            if ledger is not None:
                marks = {row["date"]: row for row in ledger["portfolio_history"]}
                if ledger.get("pre_settlement"):
                    marks[ledger["pre_settlement"]["date"]] = ledger["pre_settlement"]
                for log in output["decisions"]:
                    context = log.get("context")
                    if context is not None:
                        mark = marks[log["date"]]
                        if not math.isclose(context["cash"], mark["cash"], abs_tol=1e-7) or context["shares"] != mark["position"]:
                            raise GateError("A decision did not observe the execution ledger's actual portfolio.")
        for key, value in output.get("strategy_state", {}).items():
            if key in (
                "cash",
                "shares",
                "initial_capital",
                "total_trades",
                "winning_trades",
            ):
                _finite(value, f"strategy {key}", 0)
            elif key == "trade_returns":
                for trade_return in value:
                    _finite(trade_return, "strategy trade return")
        return {
            "rows_processed": len(source),
            "signals": len(output["signals"]),
            "decisions": len(output["decisions"]),
        }
    if stage == Stage.SIMULATE:
        history = output["portfolio_history"]
        if not history or len(history) != previous["rows_processed"]:
            raise GateError("Simulation did not cover the full data period.")
        if [record["date"] for record in history] != previous["timestamps"]:
            raise GateError("Simulation changed the analyzed data period.")
        for record in history:
            cash = _finite(record["cash"], "cash", -0.000001)
            stock = _finite(record["stock_value"], "stock value", 0)
            total = _finite(record["total_value"], "portfolio value", 0)
            _finite(record["position"], "share holdings", 0)
            if not math.isclose(cash + stock, total, rel_tol=1e-9, abs_tol=0.01):
                raise GateError("Portfolio accounting does not reconcile.")
        for trade in output["trades"]:
            if trade["timestamp"] not in previous["timestamps"]:
                raise GateError(
                    "An executed trade is outside the analyzed data period."
                )
            _finite(trade["price"], "execution price", 0.000000001)
            _finite(trade["shares"], "executed shares", 1)
            _finite(trade["commission"], "commission", 0)
            if trade["status"] != "executed" or trade["order_type"] not in (
                "buy",
                "sell",
            ):
                raise GateError(
                    "Simulation reported an unexecuted or unsupported trade."
                )
        if "execution_settings" in previous:
            from app.backtesting.ledger import validate_accounting
            try:
                validate_accounting(output, previous)
            except (ValueError, KeyError, TypeError) as exc:
                raise GateError(str(exc)) from exc
        return {
            "observations": len(history),
            "executed_trades": len(output["trades"]),
            "final_value": history[-1]["total_value"],
        }
    if stage == Stage.REVIEW:
        result = output["results"]
        metrics = result["performance_metrics"]
        for key, value in metrics.items():
            _finite(value, key)
        last = previous["portfolio_history"][-1]
        if not math.isclose(metrics["final_value"], last["total_value"], abs_tol=0.01):
            raise GateError("Reviewed final value differs from the simulation ledger.")
        if not math.isclose(
            metrics["total_return"], last["cumulative_return"], abs_tol=1e-9
        ):
            raise GateError("Reviewed return differs from the simulation ledger.")
        if metrics["num_trades"] != len(previous["trades"]):
            raise GateError("Reviewed trade count differs from the simulation ledger.")
        if not 0 <= metrics["win_rate"] <= 1 or not -1 <= metrics["max_drawdown"] <= 0:
            raise GateError("Reviewed win rate or drawdown is outside its valid range.")
        return {
            "accounting_verified": True,
            "final_value": metrics["final_value"],
            "trades": metrics["num_trades"],
        }
    if stage == Stage.REPORT:
        if output["results"] != previous["results"] or not isinstance(
            output["view"], dict
        ):
            raise GateError("Report does not match the reviewed result.")
        result, view = previous["results"], output["view"]
        if view != result:
            expected = {
                "trades": result["trades"],
                "signals": result["trading_signals"],
                "performance": result["performance_metrics"],
                "strategy_statistics": result["strategy_statistics"],
            }
            if any(view.get(k) != value for k, value in expected.items()):
                raise GateError(
                    "Published report fields differ from reviewed evidence."
                )
            statistics = view.get("statistics", {})
            strategy = result["strategy_statistics"]
            for key, expected_value in {
                "total_trades": strategy["total_trades"],
                "win_rate": strategy["strategy_win_rate"] * 100,
                "total_return": strategy["cumulative_trade_return_rate"] * 100,
                "total_realized_pnl": strategy["total_realized_pnl"],
            }.items():
                if statistics.get(key) != expected_value:
                    raise GateError(f"Published {key} differs from reviewed evidence.")
        return {"reviewed_result_preserved": True}
    raise GateError(f"Unknown stage: {stage}")


class GateStore:
    """Keep run status and content-hashed artifacts in a local SQLite database."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or settings.PIPELINE_DB_PATH
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stage_runs (
                    run_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stage_artifacts (
                    run_id TEXT NOT NULL, stage TEXT NOT NULL,
                    body TEXT NOT NULL, sha256 TEXT NOT NULL,
                    PRIMARY KEY (run_id, stage)
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def request_stop(self, run_id: str) -> dict:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT snapshot FROM stage_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            state = json.loads(row[0])
            if state["status"] not in ("completed", "failed", "cancelled"):
                state["stop_requested"] = True
                conn.execute(
                    "UPDATE stage_runs SET snapshot=? WHERE run_id=?",
                    (_canonical(state), run_id),
                )
        return self.snapshot(run_id)

    def snapshot(self, run_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT snapshot FROM stage_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        state = json.loads(row[0])
        for record in state["stages"]:
            if record["status"] == "completed":
                body = self.artifact(run_id, Stage(record["stage"]))
                if (
                    hashlib.sha256(_canonical(body).encode()).hexdigest()
                    != record["sha256"]
                ):
                    raise GateError(
                        "Saved stage status does not match its handoff evidence."
                    )
        return state

    def artifact(self, run_id: str, stage: Stage) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT body, sha256 FROM stage_artifacts WHERE run_id=? AND stage=?",
                (run_id, stage.value),
            ).fetchone()
        if row is None:
            raise GateError(f"Missing {stage.value} handoff artifact.")
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise GateError(f"Saved {stage.value} artifact failed its integrity check.")
        return json.loads(row[0])


class StageGateController:
    """Advance only in the declared sequence after validating and saving output."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        callback: Optional[Callable] = None,
        store: Optional[GateStore] = None,
        request: Optional[dict] = None,
    ):
        self.run_id = run_id or f"backtest-{uuid4().hex}"
        self.store = store or GateStore()
        self.callback = callback
        self._thread = threading.get_ident()
        self._scope = None
        self._next = 0
        self.active = None
        self.state = {
            "run_id": self.run_id,
            "status": "pending",
            "active_stage": None,
            "thread_budget": settings.PIPELINE_NUM_THREADS,
            "request": json_value(request or {}),
            "stages": [{"stage": s.value, "status": "pending"} for s in STAGES],
            "error": None,
            "stop_requested": False,
        }
        with self.store.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO stage_runs VALUES (?, ?)",
                    (self.run_id, _canonical(self.state)),
                )
            except sqlite3.IntegrityError as exc:
                raise GateError(
                    "This run ID already exists; start a new run to preserve its evidence."
                ) from exc

    def _save(self, artifact=None):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            saved = conn.execute(
                "SELECT snapshot FROM stage_runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            self.state["stop_requested"] = self.state["stop_requested"] or json.loads(
                saved[0]
            ).get("stop_requested", False)
            if artifact:
                stage, body, digest = artifact
                conn.execute(
                    "INSERT INTO stage_artifacts VALUES (?, ?, ?, ?)",
                    (self.run_id, stage.value, body, digest),
                )
            conn.execute(
                "UPDATE stage_runs SET snapshot=? WHERE run_id=?",
                (_canonical(self.state), self.run_id),
            )
        if self.callback:
            try:
                self.callback(self.store.snapshot(self.run_id))
            except Exception:
                logging.getLogger(__name__).exception(
                    "Stage status notification failed"
                )

    def start(self, stage: Stage):
        if threading.get_ident() != self._thread:
            raise GateError("Stage ownership cannot be transferred to another thread.")
        if (
            self.state["status"] in ("failed", "completed", "cancelled")
            or self.active is not None
        ):
            raise GateError("This run cannot start another stage.")
        if self._next >= len(STAGES) or stage != STAGES[self._next]:
            raise GateError(
                f"Stage order violation: expected {STAGES[self._next].value}."
            )
        if self._next:
            self.store.artifact(self.run_id, STAGES[self._next - 1])
        if self.store.snapshot(self.run_id).get("stop_requested"):
            self._cancel()
        self.state.update(status="waiting", active_stage=stage.value)
        self.state["stages"][self._next]["status"] = "waiting"
        self._save()
        self._scope = stage_resources(stage.value)
        self._scope.__enter__()
        if self.store.snapshot(self.run_id).get("stop_requested"):
            self._release()
            self._cancel()
        self.active = stage
        self.state["status"] = "running"
        self.state["stages"][self._next]["status"] = "running"
        self._save()

    def complete(self, stage: Stage, output: dict):
        if threading.get_ident() != self._thread or stage != self.active:
            raise GateError("Only the controller's active stage can complete.")
        previous = (
            self.store.artifact(self.run_id, STAGES[self._next - 1])
            if self._next
            else None
        )
        wire = json_value(output)
        try:
            evidence = validate_output(stage, wire, previous)
        except (KeyError, TypeError, ValueError) as exc:
            raise GateError(f"Invalid {stage.value} handoff: {exc}") from exc
        body = _canonical(wire)
        digest = hashlib.sha256(body.encode()).hexdigest()
        before = deepcopy(self.state)
        self.state["stages"][self._next].update(
            status="completed", sha256=digest, evidence=evidence
        )
        self.state.update(
            status="completed" if self._next + 1 == len(STAGES) else "running",
            active_stage=None,
        )
        try:
            self._save((stage, body, digest))
            self._next += 1
        except Exception:
            self.state = before
            raise
        finally:
            self._release()

    def _release(self):
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None
        self.active = None

    def _cancel(self):
        self.state.update(status="cancelled", active_stage=None, stop_requested=True)
        for record in self.state["stages"][self._next :]:
            record["status"] = "blocked"
        self._save()
        raise GateCancelled("Stopped at a stage boundary; completed evidence is saved.")

    def fail(self, error: Exception):
        if self.state["status"] in ("failed", "completed", "cancelled"):
            return
        self.state.update(status="failed", error=str(error))
        if self._next < len(STAGES):
            self.state["stages"][self._next]["status"] = "failed"
            for stage in self.state["stages"][self._next + 1 :]:
                stage["status"] = "blocked"
        try:
            self._save()
        finally:
            self._release()
