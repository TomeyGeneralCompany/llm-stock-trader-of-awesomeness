"""Persistent, paper-only autonomous trader. Never submits broker orders."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import json
import math
from pathlib import Path
import sqlite3
import time
import threading

from filelock import FileLock
import pandas as pd

from app.llm.client import get_llm_client
from app.llm.decisions import EXPLANATION_INSTRUCTIONS, DecisionError, execution_action, validate_decision
from app.llm.memory import prompt_budget
from app.paper_market import Bar, Quote, EASTERN, UTC, BAR_MINUTES, listed_symbols, market_close, read_replay, regular, utc, yahoo_bars
from app.stage_resources import stage_resources

OBJECTIVE = 100.0
START = 1.0
SHARE_STEP = Decimal("0.000001")


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True)


def number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid " + label)
    return float(value)


def features(bars):
    """Only completed bars supplied by the controller; no future rows."""
    if len(bars) < 35:
        return None
    frame = pd.DataFrame([b.record() for b in bars])
    close = frame.close
    returns = close.pct_change().dropna()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal
    delta = close.diff()
    gains = delta.clip(lower=0).tail(14).mean()
    losses = -delta.clip(upper=0).tail(14).mean()
    rsi = 100 if losses == 0 and gains > 0 else 50 if losses == 0 else 100 - 100 / (1 + gains / losses)
    change = float(close.iloc[-1] / close.iloc[-7] - 1)
    volatility = float(returns.tail(30).std(ddof=0))
    volume_baseline = float(frame.volume.iloc[-21:-1].mean())
    relative_volume = float(frame.volume.iloc[-1] / volume_baseline) if volume_baseline else 0.0
    return {
        "current": {"close": float(close.iloc[-1]), "volume": float(frame.volume.iloc[-1]),
                    "macd": float(macd.iloc[-1]), "macd_signal": float(signal.iloc[-1]), "rsi": float(rsi)},
        "technical_facts": {"return_6_bars": change, "volatility_per_bar": volatility,
                            "relative_volume": relative_volume,
                            "resistance": float(frame.high.iloc[-21:-1].max()),
                            "support": float(frame.low.iloc[-21:-1].min())},
        "trend": {"market_phase": "uptrend" if change > .003 else "downtrend" if change < -.003 else "sideways"},
        "macd_alignment": "above" if diff.iloc[-1] > 0 else "below" if diff.iloc[-1] < 0 else "equal",
        "macd_cross": ("bullish_cross" if diff.iloc[-2] <= 0 < diff.iloc[-1] else
                       "bearish_cross" if diff.iloc[-2] >= 0 > diff.iloc[-1] else "none"),
        "score": abs(change) + volatility + min(relative_volume, 10) * .002,
    }


class LocalDecision:
    """Use the existing Ollama client and semantic gate, plus measurable forecasts."""
    def __init__(self):
        self.client = get_llm_client(max_tokens=1800, response_format="json")

    def __call__(self, context, lessons):
        from app.llm.paper_schema import paper_decision_schema
        self.client.config.response_schema = paper_decision_schema(context)
        instructions = (
            "You are the local paper-trading analyst. Pursue the portfolio's stated objective. "
            "Compare alternatives and actual prior outcomes. Never assume future facts or fills. "
            "Respond with JSON action BUY/SELL/HOLD, confidence 0..1, risk_level low/medium/high, "
            "reasoning, expected_outcome, explanation, allocation_fraction 0..1 (fraction of available cash for BUY "
            "or existing shares for SELL, zero for HOLD), expected_return (signed fractional price change), "
            "horizon_bars integer 1..12, watch_conditions (list of up to 3 objects: key numeric controller field, "
            "op >= or <=, threshold number, required_position flat/long/any). For HOLD supply at least one watch condition. "
            "Watch conditions prompt fresh inference, never execute orders. "
            "Evidence keys must refer to this controller state. "
            + EXPLANATION_INSTRUCTIONS + "\nController state:\n" + dump(context)
        )
        budget = prompt_budget(self.client.config.num_ctx, self.client.config.max_tokens)
        # Complete evidence stays archived; optional context yields to account facts.
        if len(instructions.encode()) > budget-450 and context.get("intelligence"):
            import copy
            compact = copy.deepcopy(context)
            for key in ("relationships", "lessons", "catalysts"):
                while compact["intelligence"].get(key) and len(instructions.encode()) > budget-450:
                    compact["intelligence"][key].pop()
                    instructions = instructions.split("\nController state:\n", 1)[0] + "\nController state:\n" + dump(compact)
        prompt = instructions + "\nPrior outcomes (historical only):\n"
        for lesson in lessons:
            line = dump(lesson) + "\n"
            if len((prompt + line).encode()) <= budget - 450:
                prompt += line
        if len(prompt.encode()) > budget - 450:
            raise ValueError("Paper decision context exceeds its budget")
        for attempt in range(2):
            try:
                raw = self.client.invoke(prompt).content
                decision = json.loads(raw)
                check_decision(decision, context)
                return decision
            except (DecisionError, ValueError, KeyError, TypeError) as exc:
                import logging
                logging.getLogger(__name__).warning("Paper decision correction: %s; response=%s", str(exc), locals().get("raw", "")[:4000])
                if attempt:
                    raise
                prompt += "\nCorrection: " + str(exc)[:260] + ". Return complete JSON."
        raise RuntimeError("Unreachable inference state")


def check_decision(decision, context):
    validate_decision(decision, context)
    fraction = number(decision.get("allocation_fraction"), 0, 1, "allocation fraction")
    number(decision.get("expected_return"), -1, 100, "forecast return")
    horizon = decision.get("horizon_bars")
    if type(horizon) is not int or not 1 <= horizon <= 12:
        raise ValueError("Forecast horizon must be 1..12 bars")
    if (decision["action"] == "HOLD") != (fraction == 0):
        raise ValueError("Trade allocation must be positive; HOLD allocation must be zero")
    watches = decision.get("watch_conditions")
    if not isinstance(watches, list) or len(watches) > 3 or (decision["action"] == "HOLD" and not watches):
        raise ValueError("Supply watch_conditions (1..3 for HOLD)")
    for rule in watches:
        if not isinstance(rule, dict) or rule.get("op") not in (">=", "<=") or rule.get("required_position") not in ("flat", "long", "any"):
            raise ValueError("Invalid watch condition")
        key = rule.get("key", "")
        if not isinstance(key, str) or not key.startswith(("current.", "technical_facts.")):
            raise ValueError("Watch key must be a numeric controller observation")
        field, name = key.split(".", 1)
        number(context.get(field, {}).get(name), -1e15, 1e15, "watch evidence")
        number(rule.get("threshold"), -1e15, 1e15, "watch threshold")
    return decision


def triggered_watches(decision, candidate, held):
    triggered = []
    for rule in decision.get("watch_conditions", []):
        if rule["required_position"] not in ("any", "long" if held else "flat"):
            continue
        field, name = rule["key"].split(".", 1)
        value = candidate.get(field, {}).get(name)
        if value is not None and (value >= rule["threshold"] if rule["op"] == ">=" else value <= rule["threshold"]):
            triggered.append({**rule, "observed": value})
    return triggered


class PaperTrader:
    """Single-writer SQLite event ledger; atomic fills, state, decisions and outcomes."""
    def __init__(self, path, decide=None, *, challenge=True, initial=10000.0,
                 spread_bps=10, slippage_bps=5, fee=0.0, min_notional=.01,
                 reuse_proceeds=None, max_candidates=3, close=None, mode="forward", intelligence=None, use_memory=True, runtime=None):
        self.intelligence = intelligence
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(self.path + ".lock", timeout=10)
        self.analysis_lock = threading.Lock()
        self.use_memory = use_memory
        self.runtime = runtime
        self.decide = decide or LocalDecision()
        self.challenge = challenge
        self.initial = START if challenge else number(initial, .01, 1e12, "capital")
        self.slippage = number(slippage_bps, 0, 1000, "slippage") / 10000
        self.cost = number(spread_bps, 0, 1000, "spread") / 20000 + self.slippage
        self.fee = number(fee, 0, 1000, "fee")
        self.minimum = number(min_notional, .000001, 1e6, "minimum notional")
        self.reuse = challenge if reuse_proceeds is None else bool(reuse_proceeds)
        self.max_candidates = max(1, int(max_candidates))
        self.close = tuple(close) if close else None
        self.mode = mode
        config = dict(challenge=challenge, initial=self.initial, cost=self.cost, fee=self.fee,
                      minimum=self.minimum, reuse_proceeds=self.reuse, mode=mode, close=list(close) if close else None)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, body TEXT);
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, body TEXT);
                CREATE TABLE IF NOT EXISTS bars (symbol TEXT, end TEXT, body TEXT, PRIMARY KEY(symbol,end));
                CREATE TABLE IF NOT EXISTS service_heartbeat (id INTEGER PRIMARY KEY, stamp TEXT);
                CREATE TABLE IF NOT EXISTS ranked_snapshot (id INTEGER PRIMARY KEY, stamp TEXT, body TEXT);
                CREATE TABLE IF NOT EXISTS bar_receipts (symbol TEXT, end TEXT, received TEXT, PRIMARY KEY(symbol,end));
                CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, day TEXT, symbol TEXT, bar TEXT,
                    submitted TEXT, body TEXT, context TEXT, UNIQUE(day,symbol,bar));
                CREATE TABLE IF NOT EXISTS outcomes (decision_id INTEGER PRIMARY KEY, observed TEXT, body TEXT);
                CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, day TEXT, stamp TEXT, kind TEXT, body TEXT);
                CREATE TABLE IF NOT EXISTS sessions (day TEXT PRIMARY KEY, body TEXT);
            """)
            existing = db.execute("SELECT body FROM config WHERE id=1").fetchone()
            if existing and json.loads(existing[0]) != config:
                raise ValueError("Database already belongs to a different paper configuration")
            db.execute("INSERT OR IGNORE INTO config VALUES(1,?)", (dump(config),))
            db.execute("INSERT OR IGNORE INTO state VALUES(1,?)", (dump(self.new_state()),))

    def connect(self):
        from contextlib import closing
        from contextlib import contextmanager

        @contextmanager
        def connection():
            with closing(sqlite3.connect(self.path, timeout=30)) as db:
                with db:
                    yield db
        return connection()

    def new_state(self):
        return dict(day=None, cash=self.initial, unsettled=0., holdings={}, pending={}, marks={},
                    cursor=0, last_poll=None, closed=False, goal_reached_at=None)

    def state(self, db):
        return json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])

    def save(self, db, state):
        db.execute("UPDATE state SET body=? WHERE id=1", (dump(state),))

    def event(self, db, state, stamp, kind, body):
        db.execute("INSERT INTO events(day,stamp,kind,body) VALUES(?,?,?,?)",
                   (state["day"], stamp.isoformat(), kind, dump(body)))

    def equity(self, state):
        return state["cash"] + state["unsettled"] + sum(
            max(0., p["shares"] * state["marks"][symbol]["price"] * (1-self.cost) - self.fee)
            for symbol, p in state["holdings"].items())

    def closing(self, day):
        close = market_close(day, self.close)
        if close is None:
            raise ValueError("No scheduled regular session")
        return datetime.fromisoformat(day).replace(hour=close[0], minute=close[1], tzinfo=EASTERN).astimezone(UTC)

    def check_goal(self, db, state, stamp):
        if not self.challenge or state["goal_reached_at"] or state["holdings"] or state["cash"] + state["unsettled"] < OBJECTIVE:
            return
        state["goal_reached_at"] = stamp.isoformat()
        for symbol, order in list(state["pending"].items()):
            self.event(db, state, stamp, "expired", {"symbol": symbol, **order, "reason": "daily objective achieved"})
        state["pending"] = {}
        self.event(db, state, stamp, "objective_achieved", {"realized_balance": state["cash"] + state["unsettled"]})

    def end_session(self, db, state, stamp):
        if state["day"] is None or state["closed"]:
            return
        for symbol, order in list(state["pending"].items()):
            self.event(db, state, stamp, "expired", {"symbol": symbol, **order})
        state["pending"] = {}
        # Value remaining inventory at observed close less estimated exit costs.
        # This is not a fabricated closing fill and may be incomplete with stale marks.
        expected = self.closing(state["day"])
        complete = all(utc(state["marks"][sym]["end"]) >= expected for sym in state["holdings"])
        seen = db.execute("SELECT 1 FROM bars WHERE end=? LIMIT 1", (expected.isoformat(),)).fetchone()
        complete = complete and bool(seen)
        equity = self.equity(state)
        summary = dict(day=state["day"], runtime=state.get("runtime"), runtime_changed=state.get("runtime_changed",False), start=self.initial if self.challenge else state.get("opening_equity", self.initial),
                       equity=equity, cash=state["cash"], unsettled=state["unsettled"], holdings=state["holdings"],
                       goal=OBJECTIVE if self.challenge else None, complete=complete,
                       achieved=bool(state["goal_reached_at"]) if self.challenge else None,
                       goal_reached_at=state["goal_reached_at"], marked_goal=(complete and equity >= OBJECTIVE) if self.challenge else None,
                       valuation="net marked equity; goal requires realized flat balance; no invented closing fills")
        db.execute("INSERT OR REPLACE INTO sessions VALUES(?,?)", (state["day"], dump(summary)))
        state["closed"] = True
        self.event(db, state, stamp, "session_closed", summary)

    def open_session(self, db, state, day, stamp):
        if state["day"] == day:
            return
        if state["day"] and day < state["day"]:
            raise ValueError("Cannot rewind an existing forward account")
        self.end_session(db, state, stamp)
        if self.challenge:
            cursor = state["cursor"]
            state.clear()
            state.update(self.new_state())
            state["cursor"] = cursor
        else:
            state["cash"] += state["unsettled"]
            state["unsettled"] = 0.
        state.update(day=day, closed=False, opening_equity=self.equity(state))
        self.event(db, state, stamp, "session_opened", {"capital": self.equity(state), "goal": OBJECTIVE if self.challenge else None})

    def ingest(self, db, bars, now):
        for bar in bars:
            if bar.end > now or not regular(bar, *(self.close or (None, None))):
                continue
            inserted = db.execute("INSERT OR IGNORE INTO bars VALUES(?,?,?)", (bar.symbol, bar.end.isoformat(), dump(bar.record())))
            if inserted.rowcount:
                db.execute("INSERT INTO bar_receipts VALUES(?,?,?)", (bar.symbol,bar.end.isoformat(),now.isoformat()))

    def history(self, db, symbol, now, limit=100):
        rows = db.execute("SELECT body FROM bars WHERE symbol=? AND end<=? ORDER BY end DESC LIMIT ?",
                          (symbol, now.isoformat(), limit)).fetchall()
        return [Bar.from_record(json.loads(row[0])) for row in reversed(rows)]

    def fill(self, db, state, bar, stamp, *, quote=None):
        order = state["pending"].get(bar.symbol)
        market_time = quote.time if quote else bar.start
        if not order or market_time < utc(order["submitted"]) or bar.day != state["day"]:
            return
        if quote and (quote.time <= utc(order["submitted"]) or quote.time >= self.closing(bar.day)):

            return
        # First eligible next-bar opening, not the decision bar. Volume is an
        # observed liquidity cap; no fill on zero volume. Orders are IOC at that bar.
        del state["pending"][bar.symbol]
        position = state["holdings"].get(bar.symbol)
        buying = order["action"] == "BUY"
        reference = (quote.ask if buying else quote.bid) if quote else bar.open
        cost = self.slippage if quote else self.cost
        price = reference * (1+cost if buying else 1-cost)
        reserved_other = sum(o.get("reserved", 0.) for o in state["pending"].values())
        if buying:
            spend = min(order["reserved"], max(0., state["cash"] - reserved_other))
            desired = max(0., spend-self.fee) / price
        else:
            desired = min(order["quantity"], position["shares"] if position else 0.)
        capacity = (quote.ask_size if buying else quote.bid_size) if quote else bar.volume*.01
        step = Decimal("1") if quote and quote.fractional_eligible is not True else SHARE_STEP
        quantity = float(Decimal(str(min(desired, capacity))).quantize(step, rounding=ROUND_DOWN))
        notional = quantity * price
        if quantity <= 0 or notional < self.minimum or (not buying and notional < self.fee):
            self.event(db, state, stamp, "unfilled", {"symbol": bar.symbol, "order": order, "reason": "liquidity/minimum/cash"})
            return
        realized = None
        if buying:
            debit = notional + self.fee
            if debit > state["cash"] + 1e-9:
                raise RuntimeError("Cash accounting violation")
            state["cash"] -= debit
            old = position or {"shares": 0., "basis": 0.}
            state["holdings"][bar.symbol] = {"shares": old["shares"]+quantity, "basis": old["basis"]+debit}
        else:
            proceeds = notional-self.fee
            basis = position["basis"] * quantity / position["shares"]
            realized = proceeds-basis
            remaining = position["shares"]-quantity
            if remaining < 1e-10:
                del state["holdings"][bar.symbol]
            else:
                state["holdings"][bar.symbol] = {"shares": remaining, "basis": position["basis"]-basis}
            state["cash" if self.reuse else "unsettled"] += proceeds
        self.event(db, state, stamp, "fill", {"symbol": bar.symbol, "action": order["action"],
                   "quantity": quantity, "price": price, "fee": self.fee, "bar_start": bar.start.isoformat(),
                   "bar_end": bar.end.isoformat(), "decision_id": order["decision_id"], "realized_pnl": realized,
                   "fill_method": "bid_ask" if quote else "completed_bar_open", "market_time": market_time.isoformat(),
                   "received": quote.received.isoformat() if quote else stamp.isoformat(),
                   "fractional_eligibility": quote.fractional_eligible if quote else "simulated"})
        self.check_goal(db, state, quote.time if quote else bar.end)

    def score_outcomes(self, db, symbol, bar, stamp):
        rows = db.execute("""SELECT d.id,d.body,d.bar,d.context,d.submitted FROM decisions d
            LEFT JOIN outcomes o ON o.decision_id=d.id
            WHERE d.symbol=? AND d.bar<? AND o.decision_id IS NULL ORDER BY d.id""",
                          (symbol, bar.end.isoformat())).fetchall()
        for ident, body, start, original, submitted in rows:
            decision, context = json.loads(body), json.loads(original)
            base = db.execute("SELECT end,body FROM bars WHERE symbol=? AND end>=? AND end<=? ORDER BY end LIMIT 1",
                              (symbol, max(start,submitted), bar.end.isoformat())).fetchone()
            if base is None:
                continue
            origin, base_body = base
            after = db.execute("SELECT end,body FROM bars WHERE symbol=? AND end>? AND end<=? ORDER BY end LIMIT ?",
                               (symbol, origin, bar.end.isoformat(), decision["horizon_bars"])).fetchall()
            if len(after) < decision["horizon_bars"]:
                continue
            observed, final = after[-1]
            actual = json.loads(final)["close"] / json.loads(base_body)["close"] - 1
            expected = decision["expected_return"]
            outcome = dict(symbol=symbol, predicted_return=expected, actual_return=actual,
                           forecast_available=submitted, reference_bar=origin,
                           error=expected-actual, direction_correct=((expected>0)-(expected<0)) == ((actual>0)-(actual<0)),
                           action=decision["action"], forecast=decision["explanation"]["forecast"]["scenario"][:180],
                           phase=context["trend"]["market_phase"])
            outcome["observation_bar"] = observed
            db.execute("INSERT INTO outcomes VALUES(?,?,?)", (ident, stamp.isoformat(), dump(outcome)))
            self.event(db, {"day": bar.day}, stamp, "forecast_scored", {"decision_id": ident, **outcome})

    def lessons(self, db, now, symbol):
        rows = db.execute("""SELECT o.body FROM outcomes o JOIN decisions d ON d.id=o.decision_id
                            WHERE o.observed<=? ORDER BY (d.symbol=?) DESC, o.observed DESC LIMIT 12""",
                          (now.isoformat(), symbol)).fetchall()
        values = [json.loads(row[0]) for row in rows]
        # Learning survives daily resets and compares both forecasts and fills.
        groups = db.execute("""SELECT json_extract(body,'$.phase'),count(*),
                    avg(abs(json_extract(body,'$.error'))), avg(json_extract(body,'$.direction_correct'))
                    FROM outcomes WHERE observed<=? GROUP BY 1 ORDER BY 2 DESC LIMIT 4""",
                    (now.isoformat(),)).fetchall()
        fills = db.execute("""SELECT body FROM events WHERE kind='fill' AND stamp<=?
                    AND json_extract(body,'$.realized_pnl') IS NOT NULL ORDER BY id DESC LIMIT 12""",
                    (now.isoformat(),)).fetchall()
        closed = [json.loads(x[0]) for x in fills]
        summary = {"forecast_patterns": [{"phase": phase, "count": count, "mean_absolute_error": error,
                    "direction_accuracy": accuracy} for phase,count,error,accuracy in groups],
                    "recent_realized_pnl": sum(x["realized_pnl"] for x in closed),
                    "recent_closed_trades": len(closed),
                    "recent_losing_trades": sum(x["realized_pnl"] < 0 for x in closed)}
        return [summary, *values] if groups or closed else []

    def latest_candidates(self, db, state, now):
        results = []
        recent = now - timedelta(minutes=15)
        rows = db.execute("SELECT symbol,MAX(end) FROM bars WHERE end<=? AND end>=? GROUP BY symbol",
                          (now.isoformat(), recent.isoformat())).fetchall()
        for symbol, end in rows:
            bars = self.history(db, symbol, now)
            if not bars or bars[-1].day != state["day"]:
                continue
            info = features(bars)
            if info is None or bars[-1].volume <= 0:
                continue
            info["symbol"] = symbol
            info["asof"] = end
            if self.intelligence is not None:
                info["score"], info["intelligence"] = self.intelligence.rank(symbol, info["score"], now, bars=bars, closing=self.closing(state["day"]))
                if not self.use_memory:
                    info["intelligence"]["lessons"] = []
            results.append(info)
        return sorted(results, key=lambda x: (-x["score"], x["symbol"]))

    def tick(self, bars, now, *, time_source=None, cursor=None, finish=False, analyze=True, quotes=()):
        now = utc(now)
        day = now.astimezone(EASTERN).date().isoformat()
        # Serialize controller even across processes; writes are recoverable.
        with self.lock:
            with self.connect() as db:
                state = self.state(db)
                if state["last_poll"] and now < utc(state["last_poll"]):
                    raise ValueError("Paper clock cannot move backwards")
                self.ingest(db, bars, now)
                db.execute("INSERT OR REPLACE INTO service_heartbeat VALUES (1,?)", (now.isoformat(),))
                actual = [b for b in bars if b.end <= now and b.day == day and regular(b, *(self.close or (None, None)))]
                if cursor is not None:
                    state["cursor"] = cursor
                if not actual and state["day"] != day:
                    if state["day"] and not state["closed"] and now >= self.closing(state["day"]):
                        self.end_session(db, state, now)
                    self.save(db, state)
                    return state
                if state["day"] != day:
                    self.open_session(db, state, day, now)
                eligible_quotes = {}
                for quote in sorted(quotes, key=lambda item: (item.time, item.received)):
                    order = state["pending"].get(quote.symbol)
                    if (order and quote.time.astimezone(EASTERN).date().isoformat() == day
                            and utc(order["submitted"]) < quote.time < self.closing(day) and quote.received <= now):
                        eligible_quotes.setdefault(quote.symbol, quote)
                if self.runtime is not None:
                    if state.get("runtime") not in (None,self.runtime):
                        state["runtime_changed"] = True
                    state["runtime"] = self.runtime
                latest = {}
                for bar in actual:
                    if bar.symbol not in latest or bar.end > latest[bar.symbol].end:
                        latest[bar.symbol] = bar
                # Process one latest observation per symbol; fill lookup and outcome
                # scoring still use all persisted intervening bars.
                for bar in sorted(latest.values(), key=lambda b: (b.end, b.symbol)):
                    # Existing orders only fill on the earliest persisted eligible bar,
                    # even if a provider omitted it from a later refresh.
                    if not state["closed"]:
                        order = state["pending"].get(bar.symbol)
                        if order:
                            earliest = next((b for b in self.history(db, bar.symbol, now, 500)
                                             if b.day == day and b.start >= utc(order["submitted"])), None)
                            quote = eligible_quotes.get(bar.symbol)
                            if quote and (earliest is None or quote.time <= earliest.start):
                                self.fill(db,state,bar,now,quote=quote)
                            elif earliest:
                                self.fill(db, state, earliest, now)
                    saved_bar = db.execute("SELECT body FROM bars WHERE symbol=? AND end=?",
                                           (bar.symbol, bar.end.isoformat())).fetchone()
                    bar = Bar.from_record(json.loads(saved_bar[0]))
                    previous = state["marks"].get(bar.symbol)
                    if not previous or bar.end > utc(previous["end"]):
                        state["marks"][bar.symbol] = {"price": bar.close, "end": bar.end.isoformat()}
                    self.score_outcomes(db, bar.symbol, bar, now)
                for symbol, quote in eligible_quotes.items():
                    if symbol not in latest and not state["closed"] and symbol in state["marks"]:
                        # An existing mark supplies the session; the bid/ask supplies the fill price.
                        previous = self.history(db,symbol,now,1)
                        if previous and previous[-1].day == day:
                            self.fill(db,state,previous[-1],now,quote=quote)
                state["last_poll"] = now.isoformat()
                closing = self.closing(day)
                closing_seen = db.execute("SELECT 1 FROM bars WHERE end=? LIMIT 1", (closing.isoformat(),)).fetchone()
                # Allow delayed providers to deliver the final completed bar.
                required = set(state["holdings"]) | set(state["pending"])
                final_marks = all(utc(state["marks"].get(sym, {"end": "1970-01-01T00:00:00+00:00"})["end"]) >= closing for sym in required)
                if finish or (now >= closing and ((closing_seen and final_marks) or now >= closing + timedelta(minutes=30))):
                    self.end_session(db, state, now)
                self.save(db, state)

        if self.intelligence is not None:
            self.intelligence.observe(bars, now)
        if analyze:
            self.analyze(now, time_source=time_source)
        with self.connect() as db:
            return self.state(db)

    def analyze(self, now, *, time_source=None, limit=None):
        """Build requests from the current ledger; infer outside its write lock."""
        if not self.analysis_lock.acquire(blocking=False):
            return 0
        try:
            return self._analyze(now, time_source, limit)
        finally:
            self.analysis_lock.release()

    def _analyze(self, now, time_source, limit):
        now = utc(now)
        day = now.astimezone(EASTERN).date().isoformat()
        completed = 0
        with self.lock, self.connect() as db:
            state = self.state(db)
            if state["day"] != day or state["closed"] or state["goal_reached_at"]:
                return 0
            closing = self.closing(day)
            if now >= closing:
                return 0
            ranked = self.latest_candidates(db, state, now)
            ranking_report = [{"symbol": x["symbol"], "bar": x["asof"], "score": x["score"],
                               **x.get("intelligence", {}).get("ranking", {}),
                               "catalysts": x.get("intelligence", {}).get("catalysts", [])} for x in ranked[:20]]
            db.execute("INSERT OR REPLACE INTO ranked_snapshot VALUES (1,?,?)", (now.isoformat(), dump(ranking_report)))
            # Prioritize open positions and unresolved conditional plans; reserve
            # capacity for discovery and order candidates by market score.
            prior = {}
            for x in ranked:
                previous = db.execute("SELECT body FROM decisions WHERE day=? AND symbol=? ORDER BY id DESC LIMIT 1",
                                      (day, x["symbol"])).fetchone()
                if previous:
                    prior[x["symbol"]] = json.loads(previous[0])
                x["triggered_watches"] = triggered_watches(prior.get(x["symbol"], {}), x, x["symbol"] in state["holdings"])
            eligible = [x for x in ranked if x["symbol"] not in state["pending"] and not db.execute(
                "SELECT 1 FROM decisions WHERE day=? AND symbol=? AND bar=?",
                (day, x["symbol"], x["asof"])).fetchone()]
            eligible.sort(key=lambda x: (x["symbol"] not in state["holdings"], not x["triggered_watches"], -x["score"]))
            selected = eligible[:limit or self.max_candidates]
            discoveries = [x for x in eligible if x["symbol"] not in prior and x not in selected]
            if discoveries and len(selected) > 1 and all(x["symbol"] in prior for x in selected):
                selected[-1] = discoveries[0]
            alternatives = [{"symbol": x["symbol"], "score": round(x["score"], 6),
                             "return_6_bars": round(x["technical_facts"]["return_6_bars"], 5),
                             "prior_forecast": prior.get(x["symbol"], {}).get("expected_return")}
                            for x in ranked[:5]]
        for candidate in selected:
            with self.lock, self.connect() as db:
                state = self.state(db)
                symbol = candidate["symbol"]
                if state["day"] != day or state["closed"] or state["goal_reached_at"]:
                    break
                if symbol in state["pending"]:
                    continue
                reserved = sum(x.get("reserved", 0.) for x in state["pending"].values())
                available = max(0., state["cash"]-reserved)
                previous = db.execute("SELECT body FROM decisions WHERE symbol=? ORDER BY id DESC LIMIT 1", (symbol,)).fetchone()
                plans = json.loads(previous[0])["explanation"]["conditional_actions"][:2] if previous else []
                # Bound retrieved plan text independently of model output size.
                plans = [{"action": p["action"], "trigger": p["trigger"][:140],
                          "required_position": p["required_position"], "timeframe": p["timeframe"][:80]} for p in plans]
                exposure = [{"symbol": sym, "value": round(pos["shares"] * state["marks"][sym]["price"], 6)}
                            for sym, pos in sorted(state["holdings"].items(), key=lambda item: -item[1]["basis"])[:4]]
                context = {**candidate, "triggered_watches": candidate["triggered_watches"][:2],
                           "observation_timeframe": "5-minute completed bars through "+candidate["asof"],
                           "shares": state["holdings"].get(symbol, {}).get("shares", 0.),
                           "portfolio": {"available_cash": available, "unsettled": state["unsettled"],
                                         "equity": self.equity(state), "holding_count": len(state["holdings"]),
                                         "largest_exposures": exposure,
                                         "symbol_basis": state["holdings"].get(symbol, {}).get("basis", 0.),
                                         "proceeds_reusable": self.reuse,
                                         "cost_fraction_each_side": self.cost, "fee": self.fee, "min_notional": self.minimum},
                           "objective": {"starting_capital": START, "target": OBJECTIVE, "deadline": closing.isoformat()}
                                        if self.challenge else {"target": "maximize net portfolio performance"},
                           "alternatives": alternatives, "prior_conditional_plans": plans}
                lessons = self.lessons(db, now, symbol) if self.use_memory else []
                if candidate["triggered_watches"]:
                    self.event(db, state, now, "watch_triggered", {"symbol": symbol, "conditions": candidate["triggered_watches"]})
            started = utc(time_source()) if time_source else now
            timer = time.monotonic()
            try:
                with stage_resources("ANALYZE"):
                    inference_started = time.monotonic()
                    decision = check_decision(self.decide(context, lessons), context)
            except Exception as exc:
                with self.lock, self.connect() as db:
                    self.event(db, self.state(db), utc(time_source()) if time_source else now,
                               "analysis_error", {"symbol": symbol, "error": str(exc)[:500]})
                continue
            submitted = utc(time_source()) if time_source else now
            with self.lock, self.connect() as db:
                state = self.state(db)
                row = db.execute("INSERT OR IGNORE INTO decisions(day,symbol,bar,submitted,body,context) VALUES(?,?,?,?,?,?)",
                                 (day, symbol, candidate["asof"], submitted.isoformat(), dump(decision), dump(context)))
                if not row.rowcount:
                    continue
                completed += 1
                ident = row.lastrowid
                self.event(db, {"day": day}, submitted, "analysis_completed",
                           {"symbol": symbol, "decision_id": ident, "bar": candidate["asof"],
                            "started": started.isoformat(), "finished": submitted.isoformat(),
                            "queue_seconds": inference_started-timer,
                            "inference_seconds": time.monotonic()-inference_started})
                action = execution_action(decision)
                if action == "HOLD":
                    continue
                current_shares = state["holdings"].get(symbol, {}).get("shares", 0.)
                available_now = max(0., state["cash"]-sum(x.get("reserved", 0.) for x in state["pending"].values()))
                reason = None
                if state["day"] != day or state["closed"] or state["goal_reached_at"] or submitted >= closing:
                    reason = "session or objective closed"
                elif submitted-utc(candidate["asof"]) > timedelta(minutes=15):
                    reason = "stale market snapshot"
                elif symbol in state["pending"] or current_shares != context["shares"]:
                    reason = "position or order changed during inference"
                elif state["marks"].get(symbol, {}).get("end", "") > candidate["asof"]:
                    reason = "newer stock snapshot arrived during inference"
                fraction = decision["allocation_fraction"]
                order = {"action": action, "submitted": submitted.isoformat(), "decision_id": ident}
                if reason is None and action == "BUY":
                    order["reserved"] = min(available, available_now)*fraction
                    if order["reserved"] < self.minimum+self.fee:
                        reason = "insufficient buying power"
                if reason is None and action == "SELL":
                    order["quantity"] = current_shares*fraction
                    if order["quantity"] <= 0:
                        reason = "no eligible holdings"
                if reason:
                    self.event(db, {"day": day}, submitted, "order_not_queued",
                               {"symbol": symbol, "decision_id": ident, "reason": reason})
                    continue
                state["pending"][symbol] = order
                self.event(db, state, submitted, "order_queued", {"symbol": symbol, **order})
                self.save(db, state)
        return completed

    def watchlist(self, now=None):
        with self.connect() as db:
            state = self.state(db)
            now = utc(now) if now else utc(state["last_poll"]) if state["last_poll"] else datetime.now(UTC)
            recent = []
            if not state["closed"]:
                rows = db.execute("""SELECT symbol,bar,body FROM decisions WHERE day=? AND id IN
                    (SELECT MAX(id) FROM decisions WHERE day=? GROUP BY symbol)""",
                    (state["day"], state["day"])).fetchall()
                for symbol, stamp, body in rows:
                    decision = json.loads(body)
                    if decision.get("watch_conditions") and now <= utc(stamp) + timedelta(minutes=BAR_MINUTES*decision["horizon_bars"]):
                        recent.append(symbol)
            return sorted(set(state["holdings"]) | set(state["pending"]) | set(recent)), state["cursor"]

    def report(self):
        return read_report(self.path)


def read_report(path):
    """Read-only status for CLI/dashboard, without creating an account or loading AI."""
    from contextlib import closing
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as db:
        db.execute("BEGIN")
        config = json.loads(db.execute("SELECT body FROM config WHERE id=1").fetchone()[0])
        state = json.loads(db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
        equity = state["cash"] + state["unsettled"] + sum(max(0., p["shares"]*state["marks"][sym]["price"]*(1-config["cost"])-config["fee"])
                                                         for sym,p in state["holdings"].items())
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        heartbeat = db.execute("SELECT stamp FROM service_heartbeat WHERE id=1").fetchone() if "service_heartbeat" in tables else None
        ranking = db.execute("SELECT stamp,body FROM ranked_snapshot WHERE id=1").fetchone() if "ranked_snapshot" in tables else None
        return {"mode": config["mode"], "challenge": config["challenge"], "profile": config, "state": state,
                "equity": equity, "buying_power_model": "immediate paper reuse" if config["reuse_proceeds"] else "next-session settlement",
                "sessions": [json.loads(r[0]) for r in db.execute("SELECT body FROM sessions ORDER BY day")],
                "events": dict(db.execute("SELECT kind,count(*) FROM events GROUP BY kind")),
                "forecasts": db.execute("SELECT count(*) FROM decisions").fetchone()[0],
                "scored": db.execute("SELECT count(*) FROM outcomes").fetchone()[0],
                "latest_decisions": [{"symbol": sym, "bar": bar, "decision": json.loads(body)}
                    for sym,bar,body in db.execute("SELECT symbol,bar,body FROM decisions ORDER BY id DESC LIMIT 8")],
                "service_checked_at": heartbeat[0] if heartbeat else None,
                "last_quote_received": db.execute("SELECT MAX(received) FROM bar_receipts").fetchone()[0]
                    if "bar_receipts" in tables else None,
                "ranked_opportunities": json.loads(ranking[1]) if ranking else [],
                "ranked_at": ranking[0] if ranking else None,
                "recent_events": [{"time": stamp, "kind": kind, "detail": json.loads(body)}
                    for stamp,kind,body in db.execute("SELECT stamp,kind,body FROM events ORDER BY id DESC LIMIT 15")]}


def replay(engine, bars, warmup_sessions=0):
    if engine.mode != "replay":
        raise ValueError("Replay requires a separate replay-mode account")
    bars = [b for b in bars if regular(b, *(engine.close or (None,None)))]
    days = sorted({bar.day for bar in bars})
    if not days or warmup_sessions < 0 or warmup_sessions >= len(days):
        raise ValueError("Replay requires at least one evaluation session beyond warmup")
    warm_days = set(days[:warmup_sessions])
    with engine.lock, engine.connect() as db:
        if engine.state(db)["last_poll"]:
            raise ValueError("Use a fresh database for each held-out replay")
        warm = [b for b in bars if b.day in warm_days]
        if warm:
            engine.ingest(db, warm, max(b.end for b in warm))
    groups = defaultdict(list)
    # Historical CSV can include preceding sessions for warmup; every decision
    # only sees observations through the current replay clock.
    for bar in bars:
        if bar.day not in warm_days:
            groups[bar.end].append(bar)
    for end, group in sorted(groups.items()):
        engine.tick(group, end)
    last = max(groups)
    engine.tick([],last,finish=True,analyze=False)
    return engine.report()


def main():
    p = argparse.ArgumentParser(description="Autonomous PAPER trader; no brokerage or real orders.")
    p.add_argument("command", choices=("run", "once", "replay", "status"))
    from app.config import settings
    p.add_argument("--db", default=settings.PAPER_DB_PATH)
    p.add_argument("--intelligence-db", help="Explicit evidence archive; replay still enforces original availability times")
    p.add_argument("--quotes", help="Optional local JSONL bid/ask feed; Yahoo bars otherwise")
    p.add_argument("--without-memory", action="store_true", help="Disable prior decision episodes for comparison")
    p.add_argument("--symbols", help="Comma-separated override; default US exchange listings")
    p.add_argument("--csv", help="Replay CSV: symbol,end,open,high,low,close,volume; end in UTC")
    p.add_argument("--warmup-sessions", type=int, default=1, help="Historical sessions loaded without inference or fills")
    p.add_argument("--normal", action="store_true", help="Use persistent normal portfolio rather than daily $1 challenge")
    p.add_argument("--capital", type=float, default=10000.)
    settlement = p.add_mutually_exclusive_group()
    settlement.add_argument("--reuse-proceeds", dest="reuse_proceeds", action="store_true", default=None,
                            help="Hypothetical immediate paper buying power")
    settlement.add_argument("--settled-cash", dest="reuse_proceeds", action="store_false", default=None,
                            help="Require next-session settlement")
    p.add_argument("--spread-bps", type=float, default=10)
    p.add_argument("--slippage-bps", type=float, default=5)
    p.add_argument("--fee", type=float, default=0.)
    p.add_argument("--min-notional", type=float, default=.01)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--close", help="Override regular US exchange calendar close (HH:MM Eastern)")
    args = p.parse_args()
    if min(args.batch, args.candidates, args.poll) < 1:
        p.error("batch, candidates and poll must be positive")
    close = tuple(map(int, args.close.split(":"))) if args.close else None
    if close and (len(close) != 2 or not ((9,30) < close <= (16,0) and 0 <= close[1] < 60)):
        p.error("Invalid regular-session close")
    if args.command == "status":
        print(json.dumps(read_report(args.db), indent=2))
        return
    intelligence = None
    if settings.INTELLIGENCE_ENABLED:
        from app.intelligence.controller import IntelligenceController
        intelligence_path = args.intelligence_db or (args.db + ".intelligence.sqlite3" if args.command == "replay" else settings.INTELLIGENCE_DB_PATH)
        intelligence = IntelligenceController(intelligence_path, settings)
    engine = PaperTrader(args.db, intelligence=intelligence, challenge=not args.normal, initial=args.capital,
                         spread_bps=args.spread_bps, slippage_bps=args.slippage_bps, fee=args.fee,
                         min_notional=args.min_notional, reuse_proceeds=args.reuse_proceeds,
                         max_candidates=args.candidates, close=close,
                         mode="replay" if args.command=="replay" else "forward", use_memory=not args.without_memory)
    if args.command == "replay":
        if not args.csv:
            p.error("replay requires --csv")
        print(json.dumps(replay(engine, read_replay(args.csv), args.warmup_sessions), indent=2))
        return
    from app.paper_live import run_live
    run_live(engine, args, intelligence)


if __name__ == "__main__":
    main()
