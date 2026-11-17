"""Forward paper service: price polling and inference make progress independently."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

from app.paper_market import EASTERN, UTC, Quote, listed_symbols, market_close, utc, yahoo_bars


def clock():
    return datetime.now(UTC)


class QuoteJournal:
    """Read a local append-only JSONL quote feed, leaving partial lines for next poll."""
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.identity = None
        self.offset = 0

    def poll(self):
        if self.path is None or not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            import os
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.offset:
                self.offset = 0
            self.identity = identity
            stream.seek(self.offset)
            result = []
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                self.offset += len(line)
                try:
                    row = json.loads(line)
                    for key in ("time", "received"):
                        value = datetime.fromisoformat(row[key].replace("Z", "+00:00"))
                        if value.tzinfo is None:
                            raise ValueError("Timestamp requires a timezone")
                        row[key] = utc(value)
                    result.append(Quote(**row))
                except (ValueError, TypeError, KeyError) as exc:
                    print(f"Skipped malformed bid/ask input: {type(exc).__name__}", flush=True)
            return result


def run_live(engine, args, intelligence):
    universe, refreshed = [], None
    portfolio_cache, discovery_cache = ({}, {}), ({}, {})
    runtime_day = None
    journal = QuoteJournal(args.quotes)

    def fetch_bars(symbols, now, cache):
        latest, retries = cache
        bars = []
        unknown = [symbol for symbol in symbols if symbol not in latest]
        if unknown:
            with engine.connect() as db:
                for symbol in unknown:
                    row = db.execute("SELECT MAX(end) FROM bars WHERE symbol=?", (symbol,)).fetchone()
                    latest[symbol] = utc(row[0]) if row[0] else None
        due = [symbol for symbol in symbols if symbol not in retries or retries[symbol]<=now]
        warm = [symbol for symbol in due if latest[symbol] and now-latest[symbol]<=timedelta(hours=18)]
        cold = [symbol for symbol in due if symbol not in warm]
        for group, start in ((cold,None),(warm,min((latest[symbol] for symbol in warm),default=None))):
            if not group:
                continue
            try:
                pulled = yahoo_bars(group, now, start=start-timedelta(minutes=5) if start else None)
            except Exception as exc:
                print("Market batch failed: "+str(exc), flush=True)
                for symbol in group:
                    retries[symbol] = now+timedelta(minutes=2)
                continue
            bars.extend(pulled)
            returned = {bar.symbol for bar in pulled}
            for symbol in set(group)-returned:
                retries[symbol] = now+timedelta(minutes=2)
            for bar in pulled:
                latest[bar.symbol] = max(latest[bar.symbol] or bar.end,bar.end)
                retries.pop(bar.symbol,None)
        return bars

    def audit():
        ledger = engine.ledger_identity
        after = intelligence.paper_audit_cursor(ledger)
        through, events = engine.audit_events(after)
        if events:
            intelligence.paper_audit({"recent_events": events}, clock(), ledger=ledger, through=through)

    def analyze(watch, window, submitted_at):
        # Research starts only when no current paper decision is eligible.
        completed = engine.analyze(clock(), time_source=clock, limit=1, submitted_at=submitted_at) if window else 0
        if intelligence is not None:
            audit()
            intelligence.cycle(priority=watch, research=not completed)

    future = None
    ready_since = None
    scan_job = None
    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-analysis") as worker,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-discovery") as discovery,
    ):
        while True:
            try:
                now = clock()
                local = now.astimezone(EASTERN)
                hours = market_close(local.date(), engine.close)
                window = bool(hours and (9,30) <= (local.hour,local.minute)
                              and now <= engine.closing(local.date().isoformat())+timedelta(minutes=31))
                if future is not None and future.done():
                    try:
                        future.result()
                    except Exception as exc:
                        print("Background paper work failed: "+str(exc), flush=True)
                    future = None
                watch = []
                if window or args.command == "once":
                    if runtime_day != local.date():
                        from app.config import settings
                        from app.paper_trial import runtime_identity
                        engine.runtime = runtime_identity(settings.OLLAMA_MODEL,args.quotes,args)
                        runtime_day = local.date()
                    watch, cursor = engine.watchlist(now)
                    with engine.connect() as db:
                        state = engine.state(db)
                    portfolio = set(state["holdings"]) | set(state["pending"])
                    # Process existing positions and orders before the broader scan.
                    bars = fetch_bars(sorted(portfolio), now, portfolio_cache)
                    state = engine.tick(bars, clock(), analyze=False, quotes=journal.poll())
                    if not universe or refreshed != local.date():
                        universe = sorted(set(args.symbols.upper().split(","))) if args.symbols else listed_symbols()
                        refreshed = local.date()
                    if not universe:
                        raise ValueError("Empty configured market universe")
                    catalysts = intelligence.priority_symbols(now) if intelligence is not None else []
                    watch = sorted(portfolio) + sorted((set(watch) | (set(catalysts) & set(universe))) - portfolio)
                    if scan_job is not None and scan_job[0].done():
                        finished_scan, next_cursor = scan_job
                        scan_job = None
                        state = engine.tick(finished_scan.result(), clock(), cursor=next_cursor,
                                            analyze=False, quotes=journal.poll())
                        cursor = state["cursor"]
                    if scan_job is None:
                        cursor %= len(universe)
                        scan = (universe+universe)[cursor:cursor+min(args.batch,len(universe))]
                        symbols = sorted((set(scan) | set(watch)) - portfolio)
                        next_cursor = (cursor+len(scan)) % len(universe)
                        if args.command == "once":
                            bars = fetch_bars(symbols, clock(), discovery_cache)
                            state = engine.tick(bars, clock(), cursor=next_cursor, analyze=False, quotes=journal.poll())
                        else:
                            scan_job = (discovery.submit(fetch_bars, symbols, clock(), discovery_cache), next_cursor)
                else:
                    engine.tick([], now, analyze=False)
                if args.command == "once":
                    engine.analyze(clock(), time_source=clock)
                    if intelligence is not None:
                        audit()
                        intelligence.cycle(priority=watch)
                    print(json.dumps(engine.report()), flush=True)
                    return
                if not window or state["day"] == local.date().isoformat():
                    if ready_since is None:
                        ready_since = time.monotonic()
                    if future is None:
                        future = worker.submit(analyze, watch, window, ready_since)
                        ready_since = None
                print(json.dumps(engine.heartbeat()), flush=True)
            except KeyboardInterrupt:
                return
            except Exception as exc:
                if args.command == "once":
                    raise
                print("Paper poll failed: "+str(exc), flush=True)
            time.sleep(args.poll)
