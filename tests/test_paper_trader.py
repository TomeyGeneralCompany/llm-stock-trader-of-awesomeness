"""Deterministic contract tests for autonomous paper trading, never real orders."""
from datetime import datetime, timedelta
import json

import pytest

from app.paper_market import Bar, EASTERN, UTC, market_close
from app.paper_trader import PaperTrader, check_decision, replay, read_report, triggered_watches


def day_bars(symbol="AAA", day="2025-03-03", n=35, price=10., growth=0.):
    start = datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=EASTERN)
    result = []
    for i in range(n):
        v = price + growth*i
        result.append(Bar(symbol, start+timedelta(minutes=5*(i+1)), v, v*1.01, v*.99, v, 100000))
    return result


def next_bar(previous, price=None, volume=100000):
    v = price if price is not None else previous.close
    return Bar(previous.symbol, previous.end+timedelta(minutes=5), v, v*1.01, v*.99, v, volume)


def response(ctx, action="HOLD", fraction=None, threshold=None):
    mode = "wait" if action == "HOLD" else "now"
    return {"action": action, "confidence": .8, "risk_level": "medium",
            "reasoning": "Observed data supports the present decision.",
            "expected_outcome": "Price may change.",
            "allocation_fraction": (0 if action=="HOLD" else 1) if fraction is None else fraction,
            "expected_return": .01, "horizon_bars": 1,
            "watch_conditions": [{"key": "current.close", "op": ">=",
                "threshold": ctx["current"]["close"] if threshold is None else threshold, "required_position": "any"}],
            "explanation": {
                "observed_state": {"timeframe": ctx["observation_timeframe"], "position": "long" if ctx["shares"] else "flat",
                    "market_phase": ctx["trend"]["market_phase"], "macd_cross": ctx["macd_cross"],
                    "macd_alignment": ctx["macd_alignment"], "evidence": ["Completed price bar."]},
                "interpretation": "Present evidence informs the decision.",
                "forecast": {"scenario": "Price may rise", "timeframe": "Next bar", "assumptions": ["Demand persists"], "invalidation": "Trend reverses"},
                "conditional_actions": [{"action": "BUY", "trigger": "Price improves", "timeframe": "Next bar",
                    "required_position": "flat", "prerequisite": "Observe fresh data", "reasoning": "New data informs entry"}],
                "risk_assessment": {"evidence": ["current.close"], "rationale": "Price exposure is measurable", "downside": "Price can fall"},
                "execution": {"mode": mode, "evidence": ["current.close"], "rationale": "Use present observations"},
            }}


def test_ranked_discovery_reserves_cash_and_never_fills_same_bar(tmp_path):
    calls = []
    def decide(ctx, lessons):
        calls.append(ctx)
        return response(ctx, "BUY", .75)
    engine = PaperTrader(tmp_path/"paper.db", decide, max_candidates=2)
    a = day_bars("AAA", growth=.01)
    b = day_bars("BBB", growth=.2)
    state = engine.tick(a+b, a[-1].end, cursor=64)
    assert calls[0]["symbol"] == "BBB"
    assert {x["symbol"] for x in calls[0]["alternatives"]} == {"AAA","BBB"}
    assert state["cash"] == 1 and not state["holdings"]
    assert len(state["pending"]) == 2
    assert sum(x["reserved"] for x in state["pending"].values()) <= 1
    assert state["cursor"] == 64
    n = [next_bar(a[-1]), next_bar(b[-1])]
    state = engine.tick(n, n[0].end)
    assert state["cash"] >= 0 and set(state["holdings"]) == {"AAA","BBB"}
    with engine.connect() as db:
        fills = [json.loads(row[0]) for row in db.execute("SELECT body FROM events WHERE kind='fill'")]
    assert len(fills) == 2
    assert all(f["bar_start"] == a[-1].end.isoformat() for f in fills)


def test_conditional_plan_reanalysis_and_learning_survive_restart(tmp_path):
    contexts = []
    def decide(ctx, lessons):
        contexts.append((ctx, lessons))
        return response(ctx, "HOLD")
    path = tmp_path/"persistent.db"
    data = day_bars()
    engine = PaperTrader(path, decide)
    first = engine.tick(data, data[-1].end)
    assert not first["pending"] and not first["holdings"]
    engine = PaperTrader(path, decide)
    next_ = next_bar(data[-1], 11)
    engine.tick([next_], next_.end)
    assert contexts[-1][0]["triggered_watches"]
    assert contexts[-1][0]["prior_conditional_plans"]
    assert contexts[-1][1][0]["forecast_patterns"]
    assert engine.report()["scored"] == 1
    assert engine.report()["events"]["watch_triggered"] == 1
    assert not engine.report()["state"]["pending"]
    engine.tick([next_], next_.end)
    assert engine.report()["forecasts"] == 2
    tomorrow = day_bars(day="2025-03-04", price=12)
    engine.tick(tomorrow, tomorrow[-1].end)
    assert contexts[-1][1][0]["forecast_patterns"]
    assert engine.report()["state"]["day"] == "2025-03-04"
    assert engine.report()["sessions"][0]["complete"] is False


@pytest.mark.parametrize("reuse", [True, False])
def test_realized_daily_goal_and_settlement(tmp_path, reuse):
    def decide(ctx, lessons):
        return response(ctx, "SELL" if ctx["shares"] else "BUY")
    engine = PaperTrader(tmp_path/"challenge.db", decide, reuse_proceeds=reuse)
    data = day_bars(price=1.)
    engine.tick(data, data[-1].end)
    buy = next_bar(data[-1], 1.)
    state = engine.tick([buy], buy.end)
    assert state["holdings"]["AAA"]["shares"] > .9
    sale = next_bar(buy, 150.)
    state = engine.tick([sale], sale.end)
    assert not state["holdings"] and not state["pending"]
    assert state["goal_reached_at"] == sale.end.isoformat()
    assert state["cash"]+state["unsettled"] >= 100
    assert (state["unsettled"] == 0) == reuse
    closing = datetime(2025,3,3,16,0,tzinfo=EASTERN)
    last = Bar("AAA", closing, 150, 151, 149, 150, 100000)
    engine.tick([last], last.end)
    engine.tick([], last.end + timedelta(minutes=30), analyze=False)
    report = read_report(engine.path)
    assert report["sessions"][0]["achieved"] and not report["sessions"][0]["complete"]
    assert report["events"]["objective_achieved"] == 1
    tomorrow = day_bars(day="2025-03-04", price=5)
    state = engine.tick(tomorrow, tomorrow[-1].end)
    assert state["cash"] == 1 and state["goal_reached_at"] is None


def test_no_future_or_stale_execution(tmp_path):
    data = day_bars()
    later = next_bar(data[-1], 1000)
    engine = PaperTrader(tmp_path/"paper.db", lambda ctx,_: response(ctx,"BUY"))
    state = engine.tick(data+[later], data[-1].end,
                        time_source=lambda: data[-1].end+timedelta(minutes=20))
    assert not state["pending"]
    with engine.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM bars WHERE end>?", (data[-1].end.isoformat(),)).fetchone()[0] == 0
    assert engine.report()["events"]["order_not_queued"] == 1


def test_zero_volume_does_not_fill_and_duplicate_poll_is_idempotent(tmp_path):
    data = day_bars()
    engine = PaperTrader(tmp_path/"paper.db", lambda ctx,_: response(ctx,"BUY"))
    engine.tick(data, data[-1].end)
    empty = next_bar(data[-1], volume=0)
    for _ in range(2):
        state = engine.tick([empty], empty.end)
    assert not state["holdings"] and not state["pending"]
    assert state["cash"] == 1
    assert engine.report()["events"]["unfilled"] == 1


def test_replay_warmup_has_no_decisions_and_is_separate(tmp_path):
    seen = []
    def decide(ctx, lessons):
        seen.append(ctx["asof"])
        return response(ctx)
    engine = PaperTrader(tmp_path/"replay.db", decide, mode="replay")
    warm = day_bars(day="2025-03-03")
    heldout = day_bars(day="2025-03-04", n=3)
    report = replay(engine, warm+heldout, warmup_sessions=1)
    assert seen and all(x.startswith("2025-03-04") for x in seen)
    assert report["forecasts"] == 3
    with pytest.raises(ValueError, match="fresh"):
        replay(engine, warm+heldout, warmup_sessions=1)
    with pytest.raises(ValueError, match="different"):
        PaperTrader(engine.path, decide, mode="forward")


def test_market_calendar_and_early_close():
    assert market_close("2025-12-25") is None
    assert market_close("2025-11-28") == (13,0)
    assert market_close("2025-03-03") == (16,0)
    assert market_close("2025-07-04") is None


def test_model_cannot_bypass_validation(tmp_path):
    def decide(ctx, lessons):
        d = response(ctx, "BUY")
        d["explanation"]["execution"]["mode"] = "wait"
        return d
    engine = PaperTrader(tmp_path/"paper.db", decide)
    bars = day_bars()
    state = engine.tick(bars, bars[-1].end)
    assert not state["pending"]
    assert engine.report()["events"]["analysis_error"] == 1


def test_empty_feed_advances_scanner_cursor(tmp_path):
    engine = PaperTrader(tmp_path/"paper.db", lambda ctx,_: response(ctx))
    now = datetime(2025,3,3,12,0,tzinfo=EASTERN)
    engine.tick([], now, cursor=64)
    assert engine.watchlist(now)[1] == 64


def test_status_endpoint_uses_configured_read_only_ledger(tmp_path, monkeypatch):
    from app.config import settings
    from app.api.v1.endpoints.paper_status import status
    from fastapi import HTTPException
    path = tmp_path/"status.db"
    monkeypatch.setattr(settings, "PAPER_DB_PATH", str(path))
    with pytest.raises(HTTPException) as err:
        status()
    assert err.value.status_code == 404 and not path.exists()
    engine = PaperTrader(path, lambda ctx,_: response(ctx))
    assert status()["equity"] == 1



def test_live_market_and_pending_fill_continue_during_inference(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()
    a = day_bars("AAA", growth=.2)
    b = day_bars("BBB", growth=.01)
    now = [a[-1].end]
    def decide(ctx, lessons):
        if ctx["symbol"] == "BBB":
            entered.set()
            assert release.wait(10)
        return response(ctx, "BUY", .5)

    engine = PaperTrader(tmp_path/"paper.db", decide, max_candidates=1)
    engine.tick(a+b, now[0])
    assert "AAA" in engine.report()["state"]["pending"]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(engine.analyze, now[0], time_source=lambda:now[0])
        try:
            assert entered.wait(5)
            next_a, next_b = next_bar(a[-1]), next_bar(b[-1])
            now[0] = next_a.end
            state = engine.tick([next_a,next_b],now[0],analyze=False)
            assert "AAA" in state["holdings"]
            assert state["marks"]["BBB"]["end"] == next_b.end.astimezone(UTC).isoformat()
        finally:
            release.set()
        assert future.result(timeout=5)==1
    report = engine.report()
    assert "BBB" not in report["state"]["pending"]
    assert any(e["detail"].get("reason")=="newer stock snapshot arrived during inference" for e in report["recent_events"])
    restored = PaperTrader(tmp_path/"paper.db",decide,max_candidates=1)
    restored.tick([next_a,next_b],now[0],analyze=False)
    assert restored.report()["events"]["fill"]==1


def test_late_forecast_uses_post_decision_reference(tmp_path):
    bars = day_bars()
    engine=PaperTrader(tmp_path/"paper.db",lambda ctx,lessons:response(ctx),max_candidates=1)
    ready = bars[-1].end+timedelta(minutes=2)
    engine.tick(bars,bars[-1].end,time_source=lambda:ready)
    base=next_bar(bars[-1],20)
    engine.tick([base],base.end,analyze=False)
    assert engine.report()["scored"]==0
    later=next_bar(base,21)
    engine.tick([later],later.end,analyze=False)
    with engine.connect() as db:
        result=json.loads(db.execute("SELECT body FROM outcomes").fetchone()[0])
    assert result["actual_return"]==pytest.approx(.05)
    assert result["forecast_available"]==ready.astimezone(UTC).isoformat()


def test_forward_quote_fill_uses_ask_and_checks_fractional_eligibility(tmp_path):
    from app.paper_market import Quote
    bars=day_bars()
    end=bars[-1].end
    engine=PaperTrader(tmp_path/"quote.db",lambda ctx,lessons:response(ctx,"BUY"),slippage_bps=0)
    engine.tick(bars,end)
    quote=Quote("AAA",end+timedelta(seconds=1),end+timedelta(seconds=2),9.9,10.1,100,100,True)
    engine.tick([],quote.received,analyze=False,quotes=[quote])
    fill=next(e["detail"] for e in engine.report()["recent_events"] if e["kind"]=="fill")
    assert fill["price"]==10.1 and fill["fill_method"]=="bid_ask"
    assert fill["market_time"]==quote.time.isoformat()
    no_fractions=PaperTrader(tmp_path/"whole.db",lambda ctx,lessons:response(ctx,"BUY"),slippage_bps=0)
    no_fractions.tick(bars,end)
    whole=Quote("AAA",quote.time,quote.received,9.9,10.1,100,100)
    no_fractions.tick([],quote.received,analyze=False,quotes=[whole])
    assert no_fractions.report()["events"]["unfilled"]==1


def test_close_only_cannot_verify_forward_session(tmp_path):
    engine = PaperTrader(tmp_path / "paper.db", decide=lambda *_: None)
    closing_bar = day_bars(n=78)[-1]
    engine.tick([closing_bar], closing_bar.end, analyze=False)
    engine.tick([], closing_bar.end + timedelta(minutes=30), analyze=False)
    row = engine.report()["sessions"][0]
    assert not row["complete"]
    assert row["coverage"]["observed_bars"] == 1
    assert len(row["coverage"]["missing_bar_ends"]) == 77
    assert row["coverage"]["heartbeat_gaps"]


@pytest.mark.parametrize("gap", [False, True])
def test_forward_coverage_survives_restart_but_late_backfill_cannot_hide_an_outage(tmp_path, gap):
    path = tmp_path / "paper.db"
    bars = day_bars(day="2025-11-28", n=42)
    engine = PaperTrader(path, decide=lambda *_: None)
    for index, bar in enumerate(bars):
        if index == 15:
            engine = PaperTrader(path, decide=lambda *_: None)
        if gap and 16 <= index < 22:
            continue
        arrivals = bars[16:23] if gap and index == 22 else [bar]
        engine.tick(arrivals, bar.end, analyze=False)
    engine.tick([], bars[-1].end + timedelta(minutes=30), analyze=False)
    session = engine.report()["sessions"][0]
    coverage = session["coverage"]
    assert coverage["expected_bars"] == coverage["observed_bars"] == 42
    assert session["complete"] is not gap
    assert bool(coverage["heartbeat_gaps"]) is gap
    assert bool(coverage["late_or_unverified_bar_ends"]) is gap
    assert not coverage["missing_bar_ends"]


def test_timing_distinguishes_market_time_receipt_and_stale_order(tmp_path):
    engine = PaperTrader(tmp_path / "paper.db", decide=lambda ctx, _: response(ctx, "BUY"))
    bars = day_bars()
    received = bars[-1].end + timedelta(minutes=2)
    finished = received + timedelta(minutes=15)
    engine.tick(bars, received, time_source=lambda: finished)
    timing = engine.report()["analysis_timing"]
    assert timing["completed"] == timing["rejected"] == 1
    assert timing["queued"] == timing["current_when_finished"] == 0
    row = timing["recent"][0]
    assert row["bar"] == bars[-1].end.astimezone(UTC).isoformat()
    assert row["market_received_at"] == received.astimezone(UTC).isoformat()
    assert row["inference_started_at"] <= row["finished"]
    assert row["feed_lag_seconds"] == 120
    assert row["reason"] == "stale market snapshot"
