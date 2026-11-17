"""Bounded public-source adapters. Model text cannot select commands or arbitrary URLs."""
import csv
import io
import ipaddress
import json
import re
import socket
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urljoin, urlencode

import httpx
from bs4 import BeautifulSoup

from .store import canonical, dump, stamp

DEFAULT_SOURCES = [
    ("sec", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom", 300, "feed"),
    ("federal_reserve", "https://www.federalreserve.gov/feeds/press_all.xml", 300, "feed"),
    ("bls", "https://www.bls.gov/feed/bls_latest.rss", 300, "feed"),
    ("bea", "https://apps.bea.gov/rss/rss.xml", 300, "feed"),
    ("treasury", "https://home.treasury.gov/news/press-releases", 900, "index"),
    ("gdelt", "https://api.gdeltproject.org/api/v2/doc/doc", 900, "gdelt"),
    ("eia", "https://www.eia.gov/rss/press_rss.xml", 1800, "feed"),
    ("nasdaq", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", 86400, "listing"),
    ("nasdaq_other", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", 86400, "listing"),
]


def public_url(url, domains):
    url = canonical(url)
    host = urlsplit(url).hostname
    if host not in domains:
        raise ValueError("Acquisition URL is outside configured source domains.")
    for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(row[4][0])
        if not address.is_global:
            raise ValueError("Private/local acquisition destinations are prohibited.")
    return url


class PublicFetcher:
    def __init__(self, user_agent, max_bytes=2000000):
        self.user_agent, self.max_bytes = user_agent, max_bytes

    def __call__(self, url, domains):
        for _ in range(4):
            url = public_url(url, domains)
            if urlsplit(url).hostname.endswith("sec.gov"):
                time.sleep(.2)
            with httpx.Client(timeout=httpx.Timeout(20, connect=5), trust_env=False, follow_redirects=False) as client:
                with client.stream("GET", url, headers={"User-Agent": self.user_agent, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, application/json, text/html, text/plain, */*;q=0.5"}) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers["location"])
                        continue
                    response.raise_for_status()
                    payload = bytearray()
                    for block in response.iter_bytes():
                        payload.extend(block)
                        if len(payload) > self.max_bytes:
                            raise ValueError("Source response exceeds configured acquisition bound.")
                    return bytes(payload)
        raise ValueError("Source redirected too many times.")


def text_of(value):
    soup = BeautifulSoup(value, "html.parser")
    for node in soup(["script", "style", "nav"]):
        node.decompose()
    return soup.get_text(" ", strip=True)


def parse_feed(payload, url, now):
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("External XML entities are prohibited.")
    root = ET.fromstring(payload)
    results = []
    for item in list(root.iter()):
        if item.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        values, link = {}, None
        for child in item:
            key = child.tag.rsplit("}", 1)[-1]
            values.setdefault(key, "".join(child.itertext()).strip())
            if key == "link":
                link = child.attrib.get("href") or child.text or link
        title = values.get("title", "")
        body = text_of(values.get("content") or values.get("description") or values.get("summary") or title)
        date = values.get("published") or values.get("pubDate") or values.get("updated")
        try:
            published = stamp(date) if date and re.match(r"\d{4}-", date) else stamp(parsedate_to_datetime(date)) if date else None
        except (ValueError, TypeError):
            published = None
        if title and link:
            results.append({"url": urljoin(url, link.strip()), "title": title, "text": title + "\n" + body,
                            "published": published, "language": values.get("language", "und")})
    if not results:
        raise ValueError("Source contained no parsable feed entries.")
    return results


class SourceAcquisition:
    def __init__(self, store, config, fetch=None):
        self.store, self.config = store, config
        self.time_source = None
        self.fetch = fetch or PublicFetcher(config.INTELLIGENCE_USER_AGENT, config.INTELLIGENCE_MAX_BYTES)
        self.register()

    def register(self):
        now = "1970-01-01T00:00:00+00:00"
        configured = list(DEFAULT_SOURCES)
        for index, source in enumerate(json.loads(self.config.INTELLIGENCE_FEEDS)):
            configured.append((source.get("id", f"rss_{index}"), source["url"], int(source.get("interval", 900 if source.get("type")=="corporate_ir" else 300)), "feed"))
        if self.config.FRED_API_KEY:
            configured += [("fred", "https://api.stlouisfed.org/fred/series/observations", 900, "fred"),
                           ("alfred", "https://api.stlouisfed.org/fred/series/observations", 900, "alfred")]
        with self.store.connect() as db:
            db.execute("INSERT OR IGNORE INTO sources(id,body,next_due) VALUES(?,?,?)", ("yahoo", dump({"url": "https://finance.yahoo.com", "kind": "market", "interval": 300}), "9999-01-01T00:00:00+00:00"))
            for ident, url, interval, kind in configured:
                body = {"url": canonical(url), "interval": max(30, interval), "kind": kind}
                db.execute("INSERT INTO sources(id,body,next_due) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body", (ident, dump(body), now))
                db.execute("""UPDATE acquisition_jobs SET status='superseded',error='Source configuration changed'
                    WHERE source=? AND status='queued' AND kind IN ('feed','index','listing','fred','alfred','gdelt')
                    AND json_extract(payload,'$.url')<>?""", (ident,body["url"]))
        for symbol in self.config.INTELLIGENCE_BENCHMARKS.split(","):
            if symbol.strip():
                self.store.security(symbol.strip(), {"company": symbol.strip(), "security_type": "benchmark"}, stamp())
                self.store.enqueue("yahoo", "market", {"symbol": symbol.strip()}, now)

    def schedule(self, now):
        now = stamp(now)
        with self.store.connect() as db:
            rows = db.execute("SELECT id,body FROM sources WHERE next_due<=? ORDER BY next_due,id", (now,)).fetchall()
            for row in rows:
                body = json.loads(row["body"])
                # Include the scheduled timestamp so repeated polls preserve history.
                db.execute("INSERT OR IGNORE INTO acquisition_jobs(source,kind,payload,status,due) VALUES(?,?,?,'queued',?)",
                           (row["id"], body["kind"], dump({"url": body["url"], "scheduled": now}), now))
                next_due = stamp(datetime.fromisoformat(now) + timedelta(seconds=body["interval"]))
                db.execute("UPDATE sources SET next_due=? WHERE id=?", (next_due, row["id"]))

    def process(self, now, max_jobs=2):
        self.schedule(now)
        now = stamp(now)
        # Expired leases survive process termination without losing queued work.
        with self.store.connect() as db:
            db.execute("UPDATE acquisition_jobs SET status='queued',lease=NULL WHERE status='running' AND lease<?", (now,))
        count = 0
        for _ in range(max_jobs):
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                counter = db.execute("SELECT value FROM intelligence_state WHERE key='schedule_counter'").fetchone()
                slot = int(counter[0]) if counter else 0
                db.execute("INSERT OR REPLACE INTO intelligence_state VALUES('schedule_counter',?)", (str(slot+1),))
                priority_row = db.execute("SELECT value FROM intelligence_state WHERE key='priority_symbols'").fetchone()
                priority = (json.loads(priority_row[0]) if priority_row else []) + self.config.INTELLIGENCE_BENCHMARKS.split(",")
                condition, order, extra = "", "due,id", ()
                if slot % 3 == 0:
                    condition = " AND kind NOT IN ('market','metadata','history')"
                elif slot % 3 == 1 and priority:
                    condition = " AND kind='market' AND json_extract(payload,'$.symbol') IN (SELECT value FROM json_each(?))"
                    order = "(SELECT MIN(key) FROM json_each(?) WHERE value=json_extract(payload,'$.symbol')),due,id"
                    encoded = dump(priority)
                    extra = (encoded, encoded)
                row = db.execute("SELECT * FROM acquisition_jobs WHERE status='queued' AND due<=?"
                                 + condition + " ORDER BY " + order + " LIMIT 1", (now, *extra)).fetchone()
                if row is None:
                    row = db.execute("SELECT * FROM acquisition_jobs WHERE status='queued' AND due<=? ORDER BY due,id LIMIT 1", (now,)).fetchone()
                if not row:
                    break
                db.execute("UPDATE acquisition_jobs SET status='running',lease=?,attempts=attempts+1 WHERE id=?",
                           (stamp(datetime.fromisoformat(now)+timedelta(minutes=5)), row["id"]))
            try:
                self.acquire(dict(row), now)
                with self.store.connect() as db:
                    if row["kind"] in ("market", "metadata"):
                        db.execute("UPDATE acquisition_jobs SET status='queued',due=?,lease=NULL,error=NULL WHERE id=?",
                                   (stamp(datetime.fromisoformat(now)+timedelta(seconds=300 if row["kind"]=="market" else 86400)), row["id"]))
                    else:
                        db.execute("UPDATE acquisition_jobs SET status='complete',lease=NULL,error=NULL WHERE id=?", (row["id"],))
                    db.execute("UPDATE sources SET last_success=?,failures=0,error=NULL WHERE id=?", (now, row["source"]))
                self.store.log("ACQUIRE", {"job": row["id"], "source": row["source"], "status": "success"}, now)
                count += 1
            except Exception as exc:
                error = str(exc).replace(self.config.FRED_API_KEY, "[redacted]") if self.config.FRED_API_KEY else str(exc)
                delay = min(3600, 30 * 2**min(row["attempts"], 7))
                with self.store.connect() as db:
                    db.execute("UPDATE acquisition_jobs SET status='queued',due=?,lease=NULL,error=? WHERE id=?",
                               (stamp(datetime.fromisoformat(now)+timedelta(seconds=delay)), error[:500], row["id"]))
                    db.execute("UPDATE sources SET failures=failures+1,error=? WHERE id=?", (error[:500], row["source"]))
                    if row["kind"] in ("metadata", "history"):
                        symbol = json.loads(row["payload"])["symbol"]
                        kinds = (row["kind"],)
                        for interval in kinds:
                            db.execute("INSERT INTO coverage VALUES(?,?,?,NULL,NULL,NULL,?) ON CONFLICT(symbol,kind) DO UPDATE SET status=excluded.status,error=excluded.error",
                                       (symbol, interval, "provider_unavailable", error[:500]))
                self.store.log("ACQUIRE", {"job": row["id"], "source": row["source"], "status": "retry", "error": error[:300]}, now)
        return count

    def acquire(self, job, now):
        payload = json.loads(job["payload"])
        if job["kind"] in ("market", "metadata", "history"):
            return self.market(payload["symbol"], job["kind"], now, payload)
        url = payload["url"]
        domains = {urlsplit(url).hostname}
        if job["source"] == "sec" and not self.config.SEC_USER_AGENT:
            raise ValueError("SEC_USER_AGENT must identify an operator/contact before SEC acquisition.")
        if job["kind"] == "gdelt":
            url += "?" + urlencode({"query": payload.get("query", "economy OR earnings OR regulation OR commodities"), "mode": "artlist", "format": "json", "maxrecords": 50})
        if job["kind"] in ("fred", "alfred"):
            params = {"api_key": self.config.FRED_API_KEY, "file_type": "json"}
            series = self.config.INTELLIGENCE_FRED_SERIES
            day = datetime.fromisoformat(now).date()
            params.update(series_id=series, output_type=1, sort_order="desc", limit=500,
                          offset=int(payload.get("offset", 0)),
                          observation_start=payload.get("observation_start", (day-timedelta(days=90)).isoformat()))
            if job["kind"] == "alfred":
                params.update(realtime_start="1776-07-04", realtime_end=payload.get("realtime_end",day.isoformat()),
                              observation_start=payload.get("observation_start",(day-timedelta(days=365*5)).isoformat()))
            url += "?" + urlencode(params)
        fetch = PublicFetcher(self.config.SEC_USER_AGENT, self.config.INTELLIGENCE_MAX_BYTES) if job["source"] == "sec" and isinstance(self.fetch, PublicFetcher) else self.fetch
        raw = fetch(url, domains)
        if self.time_source is not None:
            now = max(stamp(now), stamp(self.time_source()))
        if job["kind"] == "article":
            soup = BeautifulSoup(raw, "html.parser")
            title = payload.get("title") or (soup.title.get_text(" ", strip=True) if soup.title else url)
            text = text_of(raw)
            if not text:
                raise ValueError("Article contained no usable text.")
            version = self.store.archive(url=payload["url"], publisher=urlsplit(url).hostname, title=title,
                text=text, published=payload.get("published", now), acquired=now, payload=raw,
                chunk_chars=self.config.MEMORY_CHUNK_CHARS)
            self.link_followup(payload, version, now)
            return
        if job["kind"] == "listing":
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="|")
            key = "Symbol" if "Symbol" in (reader.fieldnames or []) else "ACT Symbol"
            if key not in (reader.fieldnames or []):
                raise ValueError("Exchange listing schema changed.")
            for record in reader:
                symbol = (record.get(key) or "").replace(".", "-")
                if record.get("Test Issue") != "N" or not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,15}", symbol):
                    continue
                name = record.get("Security Name", symbol)
                self.store.security(symbol, {"company": name, "exchange": record.get("Exchange", "NASDAQ"), "security_type": "ETF" if record.get("ETF") == "Y" else "listed security", "source": payload["url"], "industry": None, "sector": None, "activities": None, "geographic_exposure": [], "capitalization": None, "related_companies": []}, now)
                with self.store.connect() as db:
                    for interval in ("5m", "1d", "metadata"):
                        db.execute("INSERT OR IGNORE INTO coverage VALUES(?,?,'pending',NULL,NULL,NULL,NULL)", (symbol, interval))
                self.store.enqueue("yahoo", "metadata", {"symbol": symbol}, now)
                self.store.enqueue("yahoo", "market", {"symbol": symbol}, now)
            return
        if job["kind"] == "index":
            # Treasury publishes an HTML release index rather than the assumed RSS URL.
            soup = BeautifulSoup(raw, "html.parser")
            items, seen = [], set()
            for link in soup.select("a[href]"):
                target = urljoin(payload["url"], link["href"])
                parts = urlsplit(target)
                title = link.get_text(" ", strip=True)
                if parts.hostname != urlsplit(payload["url"]).hostname or not parts.path.startswith("/news/press-releases/") or not title or target in seen:
                    continue
                seen.add(target)
                items.append({"url":target,"title":title,"text":title,"published":None,"language":"en"})
            if not items:
                raise ValueError("Official release index contained no parsable release links.")
        elif job["kind"] == "gdelt":
            items = [{"url": item["url"], "title": item.get("title", ""), "text": item.get("title", ""), "published": None, "language": item.get("language", "und")} for item in json.loads(raw).get("articles", [])]
        elif job["kind"] in ("fred", "alfred"):
            self.archive_economic(job, payload, params, raw, now)
            return
        else:
            items = parse_feed(raw, payload["url"], now)
        for item in items:
            version = self.store.archive(**item, publisher=urlsplit(item["url"]).hostname, acquired=now,
                               payload=raw, chunk_chars=self.config.MEMORY_CHUNK_CHARS)
            self.link_followup(payload, version, now)
            link_host = urlsplit(item["url"]).hostname
            # A public source's own article is a structured follow-up, never a shell task.
            if job["kind"] in ("feed", "gdelt", "index") and urlsplit(item["url"]).scheme == "https":
                self.store.enqueue(job["source"], "article", {"url": canonical(item["url"]), "published": item["published"], "title": item["title"], "summary_hash": __import__("hashlib").sha256(item["text"].encode()).hexdigest(), **({"task": payload["task"]} if payload.get("task") else {})}, now)

    def archive_economic(self, job, payload, params, raw, now):
        import math
        from datetime import date
        result = json.loads(raw)
        rows = result.get("observations")
        if not isinstance(rows,list) or result.get("error_code"):
            raise ValueError("Economic provider did not return observations")
        source, series = job["kind"], params["series_id"]
        observations = []
        for row in rows:
            if row["value"] == ".":
                continue
            observed, start, end = (date.fromisoformat(row[key]).isoformat()
                                     for key in ("date","realtime_start","realtime_end"))
            value = float(row["value"])
            if not math.isfinite(value) or start>end:
                raise ValueError("Invalid economic observation")
            observations.append((observed,start,end,value))
        safe_query = {key: value for key,value in params.items() if key!="api_key"}
        public = payload["url"]+"?"+urlencode(safe_query)
        if observations:
            text = "\n".join(f"{series} {day}: {value} (vintage {start} through {end})"
                             for day,start,end,value in observations)
            version = self.store.archive(url=public, publisher="Federal Reserve Bank of St. Louis",
                title=f"{source.upper()} {series} observations",text=text,published=None,acquired=now,
                payload=raw,chunk_chars=self.config.MEMORY_CHUNK_CHARS,language="en")
            with self.store.connect() as db:
                db.executemany("INSERT OR IGNORE INTO economic_observations VALUES(?,?,?,?,?,?,?,?)",
                               [(source,series,*row,now,version) for row in observations])
            self.link_followup(payload,version,now)
        offset = params["offset"]+len(rows)
        if rows and offset<int(result.get("count",offset)):
            follow = {**payload, "offset":offset, "observation_start":params["observation_start"]}
            if source=="alfred":
                follow["realtime_end"] = params["realtime_end"]
            self.store.enqueue(job["source"],source,follow,now)

    def link_followup(self, payload, version, now):
        if not payload.get("task"):
            return
        with self.store.connect() as db:
            task = db.execute("SELECT event FROM research_tasks WHERE id=?", (payload["task"],)).fetchone()
            if task:
                db.execute("INSERT OR IGNORE INTO event_evidence VALUES(?,?,'unassessed')", (task["event"],version))
                db.execute("UPDATE research_tasks SET status='queued',updated=? WHERE id=? AND status='waiting_evidence'", (now,payload["task"]))

    def market(self, symbol, kind, now, payload=None):
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        if kind == "metadata":
            info = ticker.get_info()
            if self.time_source is not None:
                now = max(stamp(now), stamp(self.time_source()))
            self.store.security(symbol, {"company": info.get("longName", symbol), "sector": info.get("sector"),
                "industry": info.get("industry"), "activities": info.get("longBusinessSummary"),
                "headquarters_country": info.get("country"), "geographic_exposure": [],
                "capitalization": info.get("marketCap"), "security_type": info.get("quoteType"), "source": "https://finance.yahoo.com/quote/" + symbol}, now)
            with self.store.connect() as db:
                db.execute("INSERT OR REPLACE INTO coverage VALUES(?,?,'success',NULL,NULL,?,NULL)", (symbol, "metadata", stamp(now)))
            return
        errors = []
        for interval in (("1d",) if kind == "history" else ("5m", "1d")):
            now = max(stamp(now), stamp(self.time_source())) if self.time_source else stamp(now)
            try:
                self.market_interval(ticker, symbol, interval, now, payload if kind == "history" else None)
            except Exception as exc:
                coverage = "history" if kind == "history" else interval
                with self.store.connect() as db:
                    db.execute("""INSERT INTO coverage VALUES(?,?,'provider_unavailable',NULL,NULL,NULL,?)
                        ON CONFLICT(symbol,kind) DO UPDATE SET status=excluded.status,error=excluded.error""",
                               (symbol, coverage, str(exc)[:500]))
                errors.append(f"{coverage}: {str(exc)[:300]}")
        if errors:
            raise ValueError("; ".join(errors))

    def market_interval(self, ticker, symbol, interval, now, history):
        from app.paper_market import EASTERN, market_close

        moment = datetime.fromisoformat(now)
        local = moment.astimezone(EASTERN)
        cursor_key = f"yahoo_incremental:{symbol}:{interval}"
        options = {}
        with self.store.connect() as db:
            if history:
                ending = datetime.fromisoformat(history["end"]).date()
                options = {"start": (ending - timedelta(days=365)).isoformat(), "end": ending.isoformat()}
            elif interval == "1d":
                day = local.date()
                while (hours := market_close(day)) is None or (
                        day == local.date() and (local.hour, local.minute) < hours):
                    day -= timedelta(days=1)
                target = datetime.combine(day, local.replace(hour=hours[0], minute=hours[1],
                                          second=0, microsecond=0).timetz())
                latest = db.execute("""SELECT MAX(end) FROM market_observations
                    WHERE symbol=? AND interval='1d' AND end<=? AND acquired<=?""",
                                    (symbol, now, now)).fetchone()[0]
                if latest and latest >= stamp(target):
                    return
                options = ({"start": (datetime.fromisoformat(latest).date() - timedelta(days=2)).isoformat()}
                           if latest else {"period": "1y"})
            else:
                row = db.execute("SELECT value FROM intelligence_state WHERE key=?", (cursor_key,)).fetchone()
                previous = datetime.fromisoformat(row[0]) if row and row[0] <= now else None
                opened = local.replace(hour=9, minute=30, second=0, microsecond=0)
                end = local.replace(minute=local.minute // 5 * 5, second=0, microsecond=0)
                hours = market_close(local.date())
                if previous and hours and opened <= previous and (local.hour, local.minute) <= hours:
                    count = int((end - previous).total_seconds() // 300)
                    if count <= 0:
                        return
                    observed = db.execute("""SELECT COUNT(DISTINCT o.end)
                        FROM market_observations o JOIN market_observation_sources s
                        USING(symbol,interval,end,hash)
                        WHERE o.symbol=? AND o.interval='5m' AND o.session='regular'
                        AND s.source IN ('paper_forward_yahoo','yahoo_extended')
                        AND o.end>? AND o.end<=? AND s.acquired<=?""",
                                          (symbol, stamp(previous), stamp(end), now)).fetchone()[0]
                    if observed == count:
                        db.execute("INSERT OR REPLACE INTO intelligence_state VALUES(?,?)",
                                   (cursor_key, stamp(end)))
                        return
                options = ({"start": previous - timedelta(minutes=5)}
                           if previous and moment - previous <= timedelta(days=5) else {"period": "5d"})

        frame = ticker.history(**options, interval=interval, prepost=True, auto_adjust=False, timeout=20)
        if self.time_source:
            now = max(now, stamp(self.time_source()))
        if frame.empty and history:
            with self.store.connect() as db:
                db.execute("""INSERT INTO coverage VALUES(?,?,'source_exhausted',NULL,NULL,?,?)
                    ON CONFLICT(symbol,kind) DO UPDATE SET status=excluded.status,last_success=excluded.last_success,error=excluded.error""",
                           (symbol, "history", now, "Provider returned no older historical observations."))
            return

        rows = []
        for date, row in frame.iterrows():
            start = date.to_pydatetime()
            local_date = start.astimezone(EASTERN)
            close = market_close(local_date.date())
            if interval == "1d":
                end = (local_date.replace(hour=close[0], minute=close[1], second=0, microsecond=0)
                       if close else start + timedelta(days=1))
                session = "daily"
            else:
                end = start + timedelta(minutes=5)
                eastern_end = end.astimezone(EASTERN)
                session = "regular" if close and (9, 30) < (eastern_end.hour, eastern_end.minute) <= close else "extended"
            if stamp(end) <= now:
                body = {key.lower(): float(row[key]) for key in ("Open", "High", "Low", "Close", "Volume")}
                rows.append((end, body, session))
        if not rows:
            raise ValueError("Provider returned no completed observations.")

        with self.store.connect() as db:
            self.store.observe_batch(symbol, interval, rows, now, connection=db, source="yahoo_extended" if interval == "5m" else "yahoo_daily")
            if interval == "5m":
                latest = max(stamp(row[0]) for row in rows)
                db.execute("""INSERT INTO intelligence_state VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value=MAX(intelligence_state.value,excluded.value)""",
                           (cursor_key, latest))
            else:
                oldest = stamp(frame.index.min().to_pydatetime())
                existing = db.execute("""SELECT 1 FROM acquisition_jobs
                    WHERE kind='history' AND json_extract(payload,'$.symbol')=? LIMIT 1""", (symbol,)).fetchone()
                if history or not existing:
                    db.execute("""INSERT OR IGNORE INTO acquisition_jobs(source,kind,payload,status,due)
                        VALUES('yahoo','history',?,'queued',?)""", (dump({"symbol": symbol, "end": oldest}), now))
