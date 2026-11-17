"""Additive, independent intelligence archive and chronological evidence retrieval."""
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

UTC = timezone.utc


def stamp(value=None):
    value = value or datetime.now(UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Evidence timestamps must be timezone-aware.")
    return value.astimezone(UTC).isoformat()


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username:
        raise ValueError("Evidence URLs must be public HTTPS URLs.")
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_") and k.lower() not in ("fbclid", "gclid")]
    return urlunsplit(("https", parts.netloc.lower(), parts.path or "/", urlencode(sorted(query)), ""))


TABLES = {
    "economic_observations": "source TEXT, series TEXT, observation_date TEXT, vintage_start TEXT, vintage_end TEXT, value REAL, acquired TEXT, version TEXT, PRIMARY KEY(source,series,observation_date,vintage_start,vintage_end,value)",
    "sources": "id TEXT PRIMARY KEY, body TEXT NOT NULL, next_due TEXT NOT NULL, last_success TEXT, failures INTEGER DEFAULT 0, error TEXT",
    "acquisition_jobs": "id INTEGER PRIMARY KEY, source TEXT, kind TEXT, payload TEXT, status TEXT, due TEXT, attempts INTEGER DEFAULT 0, lease TEXT, error TEXT, UNIQUE(source,kind,payload)",
    "securities": "symbol TEXT PRIMARY KEY, body TEXT, updated TEXT",
    "security_versions": "symbol TEXT, available TEXT, body TEXT, hash TEXT, PRIMARY KEY(symbol,available,hash)",
    "market_observations": "symbol TEXT, interval TEXT, end TEXT, acquired TEXT, session TEXT, body TEXT, hash TEXT, PRIMARY KEY(symbol,interval,end,hash)",
    "documents": "id TEXT PRIMARY KEY, url TEXT UNIQUE, publisher TEXT, title TEXT, event_id TEXT",
    "source_payloads": "id TEXT PRIMARY KEY, acquired TEXT, body BLOB, hash TEXT",
    "document_versions": "id TEXT PRIMARY KEY, document TEXT, published TEXT, acquired TEXT, revised TEXT, available TEXT, event_time TEXT, hash TEXT, original_text TEXT, original_payload BLOB, language TEXT, UNIQUE(document,hash)",
    "document_chunks": "id TEXT PRIMARY KEY, version TEXT, ordinal INTEGER, start INTEGER, end INTEGER, body TEXT, hash TEXT",
    "entities": "id TEXT PRIMARY KEY, name TEXT, kind TEXT, symbol TEXT, evidence TEXT, available TEXT",
    "entity_relationships": "id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT, evidence TEXT, available TEXT",
    "events": "id TEXT PRIMARY KEY, title TEXT, classification TEXT, first_available TEXT",
    "event_evidence": "event TEXT, version TEXT, stance TEXT, PRIMARY KEY(event,version)",
    "research_tasks": "id TEXT PRIMARY KEY, event TEXT, question TEXT, status TEXT, priority REAL, updated TEXT, lease TEXT, steps INTEGER DEFAULT 0, dependencies TEXT, error TEXT",
    "research_findings": "id TEXT PRIMARY KEY, task TEXT, available TEXT, body TEXT, raw_response TEXT",
    "causal_hypotheses": "id TEXT PRIMARY KEY, event TEXT, available TEXT, body TEXT, evidence TEXT",
    "market_forecasts": "id TEXT PRIMARY KEY, hypothesis TEXT, symbol TEXT, available TEXT, target TEXT, base_price REAL, body TEXT, status TEXT",
    "forecast_outcomes": "forecast TEXT PRIMARY KEY, observed TEXT, body TEXT",
    "learned_lessons": "id TEXT PRIMARY KEY, symbol TEXT, available TEXT, body TEXT",
    "coverage": "symbol TEXT, kind TEXT, status TEXT, earliest TEXT, latest TEXT, last_success TEXT, error TEXT, PRIMARY KEY(symbol,kind)",
    "intelligence_events": "id INTEGER PRIMARY KEY, stamp TEXT, stage TEXT, body TEXT",
    "market_research_waits": "hypothesis TEXT, symbol TEXT, status TEXT, task TEXT, PRIMARY KEY(hypothesis,symbol)",
    "intelligence_state": "key TEXT PRIMARY KEY, value TEXT",
    "research_runs": "id TEXT PRIMARY KEY, task TEXT, available TEXT, prompt TEXT, response TEXT, status TEXT, error TEXT",
    "paper_audit": "id TEXT PRIMARY KEY, available TEXT, body TEXT",
}


def classify_event(value):
    text = value.lower()
    categories = [
        ("bankruptcy", ("bankrupt", "insolvency")), ("merger", ("merger", "acquisition", "takeover")),
        ("regulation", ("regulat", "antitrust")), ("legislation", ("legislat", "bill signed")),
        ("rates", ("interest rate", "federal funds", "monetary policy", "rate hike", "rate cut")),
        ("labor", ("strike", "labor", "layoff")), ("conflict", ("conflict", "war ", "sanction")),
        ("shipping", ("shipping", "freight", "port closure")), ("disaster", ("earthquake", "hurricane", "flood", "wildfire")),
        ("shortage", ("shortage", "scarcity")), ("earnings", ("earnings", "quarterly results")),
        ("launch", ("launch", "unveil")), ("resignation", ("resign", "steps down")),
        ("technology", ("technology", "semiconductor")), ("demand", ("demand", "orders", "order growth", "order increase")),
    ]
    for category, terms in categories:
        if category == text or any(term in text for term in terms):
            return category
    return text if text in {item[0] for item in categories} else "unclassified"


class IntelligenceStore:
    def __init__(self, path):
        self.path = str(path)
        self.read_only = False
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            for name, fields in TABLES.items():
                db.execute(f"CREATE TABLE IF NOT EXISTS {name} ({fields})")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(chunk UNINDEXED,body)")
            db.execute("CREATE INDEX IF NOT EXISTS evidence_availability ON document_versions(available)")
            db.execute("CREATE INDEX IF NOT EXISTS economic_asof ON economic_observations(series,acquired,observation_date,vintage_start)")
            db.execute("CREATE INDEX IF NOT EXISTS observation_time ON market_observations(symbol,end,acquired)")
            db.execute("CREATE INDEX IF NOT EXISTS tasks_queue ON research_tasks(status,priority,updated)")
            if db.execute("PRAGMA user_version").fetchone()[0] < 1:
                db.execute("PRAGMA user_version=1")

    @contextmanager
    def connect(self):
        db = sqlite3.connect("file:"+self.path+"?mode=ro", uri=True, timeout=30) if getattr(self, "read_only", False) else sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def log(self, stage, body, now):
        import logging
        logging.getLogger(__name__).info("%s %s", stage, dump(body))
        with self.connect() as db:
            db.execute("INSERT INTO intelligence_events(stamp,stage,body) VALUES(?,?,?)", (stamp(now), stage, dump(body)))

    def security(self, symbol, metadata, now):
        with self.connect() as db:
            row = db.execute("SELECT body FROM securities WHERE symbol=?", (symbol,)).fetchone()
            current = db.execute("SELECT updated FROM securities WHERE symbol=?", (symbol,)).fetchone()
            body = json.loads(row[0]) if row and current["updated"]<=stamp(now) else {}
            body.update({key: value for key, value in metadata.items() if value is not None and value != [] or key not in body})
            body["symbol"] = symbol
            encoded = dump(body)
            db.execute("INSERT OR IGNORE INTO security_versions VALUES(?,?,?,?)", (symbol, stamp(now), encoded, sha(encoded)))
            db.execute("INSERT INTO securities VALUES(?,?,?) ON CONFLICT(symbol) DO UPDATE SET body=excluded.body,updated=excluded.updated WHERE excluded.updated>=securities.updated", (symbol, encoded, stamp(now)))
            ident = sha("company:" + symbol)
            db.execute("INSERT OR IGNORE INTO entities VALUES(?,?,?,?,?,?)",
                       (ident, body.get("company", symbol), "company", symbol, "[]", stamp(now)))

    def catalog(self, now):
        with self.connect() as db:
            rows = db.execute("""SELECT symbol,body FROM (
                SELECT symbol,body,ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY available DESC,rowid DESC) rank
                FROM security_versions WHERE available<=?) WHERE rank=1""", (stamp(now),)).fetchall()
        return {row["symbol"]: json.loads(row["body"]) for row in rows}

    def observe(self, symbol, interval, end, body, acquired, session="regular"):
        end, acquired = stamp(end), stamp(acquired)
        if end > acquired:
            raise ValueError("Cannot archive an unfinished market observation.")
        for key in ("open", "high", "low", "close", "volume"):
            value = body[key]
            import math
            if not math.isfinite(value) or value < 0 or (key != "volume" and value == 0):
                raise ValueError("Invalid market observation.")
        encoded = dump(body)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO market_observations VALUES(?,?,?,?,?,?,?)",
                       (symbol, interval, end, acquired, session, encoded, sha(encoded)))
            db.execute("""INSERT INTO coverage VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(symbol,kind) DO UPDATE SET status='success',
                earliest=CASE WHEN coverage.earliest IS NULL THEN excluded.earliest ELSE MIN(coverage.earliest,excluded.earliest) END,
                latest=CASE WHEN coverage.latest IS NULL THEN excluded.latest ELSE MAX(coverage.latest,excluded.latest) END,
                last_success=excluded.last_success,error=NULL""", (symbol, interval, "success", end, end, acquired, None))

    def observations(self, symbol, now, interval="5m", limit=5000):
        with self.connect() as db:
            rows = db.execute("""SELECT end,body,session FROM (
                SELECT end,body,session,ROW_NUMBER() OVER(PARTITION BY end ORDER BY acquired DESC,rowid DESC) rank
                FROM market_observations WHERE symbol=? AND interval=? AND end<=? AND acquired<=?)
                WHERE rank=1 ORDER BY end DESC LIMIT ?""",
                (symbol, interval, stamp(now), stamp(now), limit)).fetchall()
        return [{"end": row["end"], "session": row["session"], **json.loads(row["body"])} for row in reversed(rows)]

    def archive(self, *, url, publisher, title, text, published, acquired, event_time=None,
                language="und", payload=None, chunk_chars=768, overlap=96):
        if chunk_chars <= overlap or overlap < 0:
            raise ValueError("Chunk size must exceed overlap.")
        url, acquired = canonical(url), stamp(acquired)
        published = stamp(published) if published else None
        available = max(published or acquired, acquired)
        doc, content_hash = sha(url), sha(text)
        version = sha(doc + content_hash)
        normalized = re.sub(r"\W+", " ", title.lower()).strip()
        with self.connect() as db:
            raw = payload if payload is not None else text.encode()
            payload_id = sha(raw)
            db.execute("INSERT OR IGNORE INTO source_payloads VALUES(?,?,?,?)", (payload_id, acquired, raw, payload_id))
            if db.execute("SELECT 1 FROM document_versions WHERE id=?", (version,)).fetchone():
                return version
            prior = db.execute("SELECT event_id FROM documents WHERE id=?", (doc,)).fetchone()
            event = prior["event_id"] if prior else sha(normalized)
            # Similar headlines cluster an event; evidence is never discarded.
            if not prior:
                for row in db.execute("SELECT id,title FROM events ORDER BY first_available DESC LIMIT 200"):
                    if SequenceMatcher(None, normalized, re.sub(r"\W+", " ", row["title"].lower()).strip()).ratio() >= .92:
                        event = row["id"]
                        break
            db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?)", (event, title, classify_event(title+" "+text[:500]), available))
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)", (doc, url, publisher, title, event))
            db.execute("INSERT INTO document_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (version, doc, published, acquired, acquired if prior else None, available,
                        stamp(event_time) if event_time else published or acquired, content_hash, text,
                        ("source_payload:"+payload_id).encode(), language))
            db.execute("INSERT INTO event_evidence VALUES(?,?,?)", (event, version, "unassessed"))
            for ordinal, start in enumerate(range(0, len(text), chunk_chars - overlap)):
                body = text[start:start + chunk_chars]
                chunk = sha(version + ":" + str(ordinal))
                db.execute("INSERT INTO document_chunks VALUES(?,?,?,?,?,?,?)",
                           (chunk, version, ordinal, start, start+len(body), body, sha(body)))
                db.execute("INSERT INTO evidence_fts VALUES(?,?)", (chunk, body))
                if start+chunk_chars >= len(text):
                    break
            task = sha(event + ":investigate")
            db.execute("""INSERT INTO research_tasks VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET status='queued',updated=excluded.updated,error=NULL""",
                (task, event, "Investigate causes, affected entities, secondary/opposing effects, contradictions and price incorporation.", "queued", 1., available, None, 0, "[]", None))
        self.log("INDEX", {"version": version, "event": event, "characters": len(text)}, acquired)
        return version

    def retrieve(self, query, now, limit=6):
        terms = re.findall(r"[A-Za-z0-9_]+", query)
        fts = " OR ".join('"' + term + '"' for term in terms[:20])
        with self.connect() as db:
            where = "v.available<=?"
            params = [stamp(now)]
            if fts:
                where += " AND c.id IN (SELECT chunk FROM evidence_fts WHERE evidence_fts MATCH ?)"
                params.append(fts)
            rows = db.execute(f"""SELECT c.*,v.available,v.published,v.acquired,v.event_time,
                d.url,d.publisher,d.title,d.event_id FROM document_chunks c
                JOIN document_versions v ON c.version=v.id JOIN documents d ON v.document=d.id
                WHERE {where} ORDER BY v.available DESC,c.ordinal LIMIT ?""", (*params, limit)).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            if sha(value["body"]) != value["hash"]:
                raise ValueError("Archived evidence failed integrity validation.")
            result.append(value)
        return result

    def citation(self, value, now):
        if not isinstance(value, dict):
            raise ValueError("Evidence citation must be an object.")
        with self.connect() as db:
            row = db.execute("""SELECT c.*,v.available FROM document_chunks c JOIN document_versions v
                ON c.version=v.id WHERE c.id=? AND v.available<=?""", (value.get("chunk_id"), stamp(now))).fetchone()
        if not row or sha(row["body"]) != row["hash"]:
            raise ValueError("Unknown, unavailable or corrupt evidence citation.")
        start, end = value.get("start"), value.get("end")
        if type(start) is not int or type(end) is not int or not row["start"] <= start < end <= row["end"]:
            raise ValueError("Citation range must refer to original-text offsets inside the selected chunk.")
        quote = row["body"][start-row["start"]:end-row["start"]]
        if value.get("quote") != quote:
            raise ValueError("Evidence quotation does not match archived original text.")
        return quote

    def enqueue(self, source, kind, payload, due):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO acquisition_jobs(source,kind,payload,status,due) VALUES(?,?,?,'queued',?)",
                       (source, kind, dump(payload), stamp(due)))

    def overview(self, now):
        with self.connect() as db:
            present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts = {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] if name in present else 0 for name in TABLES}
            sources = [dict(r) for r in db.execute("SELECT id,last_success,next_due,failures,error FROM sources")]
            coverage = [dict(r) for r in db.execute("SELECT status,COUNT(*) count FROM coverage GROUP BY status")]
            details = [dict(r) for r in db.execute("SELECT symbol,kind,status,earliest,latest FROM coverage ORDER BY last_success DESC LIMIT 25")]
            details += [dict(r) for r in db.execute("""SELECT symbol,interval kind,session,'observed' status,
                MIN(end) earliest,MAX(end) latest FROM market_observations WHERE symbol IN
                (SELECT symbol FROM coverage ORDER BY last_success DESC LIMIT 25)
                GROUP BY symbol,interval,session ORDER BY latest DESC LIMIT 25""")]
            tasks = [dict(r) for r in db.execute("SELECT id,question,status,priority,steps,error FROM research_tasks ORDER BY priority DESC,updated DESC LIMIT 20")]
            forecasts = [dict(r) for r in db.execute("SELECT id,symbol,available,target,status,body FROM market_forecasts ORDER BY available DESC LIMIT 20")]
            lessons = [dict(r) for r in db.execute("SELECT symbol,available,body FROM learned_lessons ORDER BY available DESC LIMIT 20")]
            logs = [dict(r) for r in db.execute("SELECT * FROM intelligence_events ORDER BY id DESC LIMIT 30")]
        return {"asof": stamp(now), "counts": counts, "sources": sources, "coverage": coverage, "coverage_details": details,
                "investigations": tasks, "forecasts": forecasts, "lessons": lessons, "activity": logs}
