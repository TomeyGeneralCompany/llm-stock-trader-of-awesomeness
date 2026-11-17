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
    universe, refreshed, seen = [], None, {}
    journal = QuoteJournal(args.quotes)

    def ingest(symbols, now, cursor=None, quotes=()):
        bars = []
        cold = [symbol for symbol in symbols if now-seen.get(symbol, datetime.min.replace(tzinfo=UTC)) > timedelta(hours=18)]
        warm = [symbol for symbol in symbols if symbol not in cold]
        for group, period in ((cold,"5d"),(warm,"1d")):
            if group:
                try:
                    bars.extend(yahoo_bars(group, now, period=period))
                except Exception as exc:
                    print("Market batch failed: "+str(exc), flush=True)
        for bar in bars:
            seen[bar.symbol] = now
        engine.tick(bars, clock(), cursor=cursor, analyze=False, quotes=quotes)

    def analyze(watch, window):
        # Research starts only when no current paper decision is eligible.
        completed = engine.analyze(clock(), time_source=clock, limit=1) if window else 0
        if intelligence is not None:
            intelligence.paper_audit(engine.report(), clock())
            intelligence.cycle(priority=watch, research=not completed)

    future = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-analysis") as worker:
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
                    watch, cursor = engine.watchlist(now)
                    with engine.connect() as db:
                        state = engine.state(db)
                    portfolio = set(state["holdings"]) | set(state["pending"])
                    # Process existing positions and orders before the broader scan.
                    ingest(sorted(portfolio), now, quotes=journal.poll())
                    if not universe or refreshed != local.date():
                        universe = sorted(set(args.symbols.upper().split(","))) if args.symbols else listed_symbols()
                        refreshed = local.date()
                    if not universe:
                        raise ValueError("Empty configured market universe")
                    catalysts = intelligence.priority_symbols(now) if intelligence is not None else []
                    watch = sorted(set(watch) | (set(catalysts) & set(universe)))
                    cursor %= len(universe)
                    scan = (universe+universe)[cursor:cursor+min(args.batch,len(universe))]
                    ingest(sorted((set(scan)|set(watch))-portfolio), clock(),
                           cursor=(cursor+len(scan))%len(universe), quotes=journal.poll())
                else:
                    engine.tick([], now, analyze=False)
                if args.command == "once":
                    engine.analyze(clock(), time_source=clock)
                    if intelligence is not None:
                        intelligence.cycle(priority=watch)
                    print(json.dumps(engine.report()), flush=True)
                    return
                if future is None:
                    future = worker.submit(analyze, watch, window)
                print(json.dumps(engine.report()), flush=True)
            except KeyboardInterrupt:
                return
            except Exception as exc:
                if args.command == "once":
                    raise
                print("Paper poll failed: "+str(exc), flush=True)
            time.sleep(args.poll)
