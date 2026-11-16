"""Continuous intelligence scheduling, learning and read-only paper-trader integration."""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from filelock import FileLock, Timeout

from app.stage_resources import stage_resources
from .store import IntelligenceStore, dump, sha, stamp, UTC
from .sources import SourceAcquisition
from .research import CausalResearch
from .statistics import score_forecast


class IntelligenceController:
    def __init__(self, path, config, *, fetch=None, client=None):
        self.store = IntelligenceStore(path)
        self.config = config
        self.lock = FileLock(str(path)+".writer.lock", timeout=0)
        self.acquisition = SourceAcquisition(self.store, config, fetch)
        self.research = CausalResearch(self.store, client)
        try:
            with self.lock, self.store.connect() as db:
                # The writer lock proves no previous cycle is still active.
                db.execute("UPDATE acquisition_jobs SET status='queued',lease=NULL WHERE status='running'")
                db.execute("UPDATE research_tasks SET status='queued',lease=NULL WHERE status='running'")
        except Timeout:
            pass

    def observe(self, bars, now):
        """Paper observations are archived, never turned into orders by research."""
        from app.paper_market import regular
        for bar in bars:
            if bar.end <= now:
                body = {key: getattr(bar, key) for key in ("open", "high", "low", "close", "volume")}
                self.store.observe(bar.symbol, "5m", bar.end, body, now, "regular" if regular(bar) else "extended")
                self.store.security(bar.symbol, {}, now)
        self.evaluate(now)

    def cycle(self, now=None, *, priority=(), acquire=True, research=True):
        now = stamp(now)
        try:
            with self.lock:
                # Holdings/pending plans can accelerate their evidence acquisition.
                with self.store.connect() as db:
                    db.execute("INSERT OR REPLACE INTO intelligence_state VALUES(?,?)", ("priority_symbols", dump(list(priority))))
                    for symbol in priority:
                        db.execute("UPDATE acquisition_jobs SET due=? WHERE kind='market' AND json_extract(payload,'$.symbol')=? AND status='queued'", (now, symbol))
                acquired = 0
                if acquire:
                    with stage_resources("ACQUIRE"):
                        acquired = self.acquisition.process(now, self.config.INTELLIGENCE_NETWORK_JOBS)
                self.evaluate(now)
                result = None
                if research:
                    try:
                        result = self.research.step(now)
                    except Exception:
                        # The durable task remains queued; acquisition and the paper
                        # controller must continue despite one analytical failure.
                        pass
                self.store.log("QUEUE", {"acquired_jobs": acquired, "research_progress": bool(result)}, now)
                return {"status": "processed", "acquired_jobs": acquired, "research_progress": bool(result)}
        except Timeout:
            return {"status": "busy"}

    def evaluate(self, now):
        now = stamp(now)
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM market_forecasts WHERE status='pending' AND target<=? AND available<=? ORDER BY target LIMIT 100", (now, now)).fetchall()
        for row in rows:
            observations = self.store.observations(row["symbol"], now)
            later = next((o for o in observations if o["end"] >= row["target"] and o["end"] > row["available"]), None)
            if later is None:
                continue
            prediction = json.loads(row["body"])
            actual = later["close"] / row["base_price"] - 1
            result = score_forecast(prediction, actual)
            result.update(forecast=row["id"], hypothesis=row["hypothesis"], target=row["target"],
                          observation=later["end"], elapsed_minutes=(datetime.fromisoformat(later["end"])-datetime.fromisoformat(row["available"])).total_seconds()/60,
                          evidence=prediction["evidence"], classification=prediction["classification"],
                          horizon_minutes=prediction["horizon_minutes"], conditions=prediction.get("conditions", "unclassified"))
            lesson_id = sha("outcome:" + row["id"])
            with self.store.connect() as db:
                db.execute("INSERT OR IGNORE INTO forecast_outcomes VALUES(?,?,?)", (row["id"], now, dump(result)))
                db.execute("UPDATE market_forecasts SET status='evaluated' WHERE id=?", (row["id"],))
                db.execute("INSERT OR IGNORE INTO learned_lessons VALUES(?,?,?,?)",
                           (lesson_id, row["symbol"], now, dump({**result, "supporting_example": bool(result["direction_correct"]),
                             "contradictory_example": not result["direction_correct"], "reliability": "one observed example; assess grouped samples",
                             "unanswered": ["What caused the discrepancy?"] if result["error_classes"] else []})))
                if result["error_classes"]:
                    task = sha("review:" + row["id"])
                    hypothesis = db.execute("SELECT event FROM causal_hypotheses WHERE id=?", (row["hypothesis"],)).fetchone()
                    db.execute("INSERT OR IGNORE INTO research_tasks VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (task, hypothesis[0] if hypothesis else "", "Investigate forecast discrepancy " + row["id"] + "; distinguish acquisition, interpretation, relationship, direction, magnitude, timing, allocation and execution causes using evidence.", "queued", 2., now, None, 0, "[]", None))
            self.store.log("MEMORY", {"forecast": row["id"], "lesson": lesson_id}, now)

    def lessons(self, symbol, now, limit=3):
        with self.store.connect() as db:
            rows = db.execute("SELECT body FROM learned_lessons WHERE symbol=? AND available<=? ORDER BY available DESC LIMIT ?", (symbol, stamp(now), limit)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def context(self, symbol, now, max_bytes=650):
        now = stamp(now)
        with self.store.connect() as db:
            rows = db.execute("""SELECT id,body,available FROM market_forecasts
                WHERE symbol=? AND available<=? AND target>=? AND status='pending' ORDER BY available DESC LIMIT 3""", (symbol, now, now)).fetchall()
            relations = db.execute("""SELECT a.name subject,r.predicate,b.name object,r.evidence,r.available
                FROM entity_relationships r JOIN entities a ON r.subject=a.id JOIN entities b ON r.object=b.id
                WHERE r.available<=? AND (a.symbol=? OR b.symbol=? OR a.name=? OR b.name=?)
                ORDER BY r.available DESC LIMIT 2""", (now, symbol, symbol, symbol, symbol)).fetchall()
        output = {"catalysts": [], "relationships": [], "lessons": []}
        for row in rows:
            value = json.loads(row["body"])
            item = {"forecast_id": row["id"], "classification": value["classification"],
                    "direction": value["direction"], "expected_return": value["expected_return"],
                    "horizon_minutes": value["horizon_minutes"], "confidence": value["confidence"],
                    "invalidation": value["invalidation"][:80], "available": row["available"],
                    "evidence": [{key: citation[key] for key in ("chunk_id", "start", "end")} for citation in value["evidence"][:1]]}
            candidate = {**output, "catalysts": output["catalysts"] + [item]}
            if len(dump(candidate).encode()) <= max_bytes:
                output = candidate
        for row in relations:
            item = {key: row[key] for key in ("subject", "predicate", "object", "available")}
            candidate = {**output, "relationships": output["relationships"] + [item]}
            if len(dump(candidate).encode()) <= max_bytes:
                output = candidate
        for lesson in self.lessons(symbol, now):
            item = {"actual_return": lesson.get("actual_return"), "absolute_error": lesson.get("absolute_error"),
                    "direction_correct": lesson.get("direction_correct"), "error_classes": lesson.get("error_classes")}
            candidate = {**output, "lessons": output["lessons"]+[item]}
            if len(dump(candidate).encode()) <= max_bytes:
                output = candidate
        return output

    def rank(self, symbol, technical_score, now):
        context = self.context(symbol, now)
        score = technical_score
        for forecast in context["catalysts"]:
            # Relevance, magnitude, confidence and freshness affect priority;
            # a favorable story is not an execution signal.
            horizon = max(1, forecast["horizon_minutes"])
            age = max(0., (datetime.fromisoformat(stamp(now))-datetime.fromisoformat(forecast["available"])).total_seconds())
            score += min(1., abs(forecast["expected_return"])) * forecast["confidence"] * math.exp(-age/(horizon*60)) / math.sqrt(horizon)
        return score, context

    def paper_audit(self, report, now):
        """Mirror reported execution errors/results without modifying the paper DB."""
        for event in report.get("recent_events", []):
            body = dump(event)
            ident = sha(body)
            with self.store.connect() as db:
                db.execute("INSERT OR IGNORE INTO paper_audit VALUES(?,?,?)", (ident, stamp(now), body))

    def overview(self, now=None):
        now = stamp(now)
        result = self.store.overview(now)
        groups = {}
        with self.store.connect() as db:
            rows = db.execute("""SELECT f.symbol,f.body forecast,o.body outcome FROM forecast_outcomes o
                JOIN market_forecasts f ON o.forecast=f.id WHERE o.observed<=? AND f.available<=?""", (now, now)).fetchall()
            metadata = self.store.catalog(now)
        for row in rows:
            forecast, outcome = json.loads(row["forecast"]), json.loads(row["outcome"])
            labels = {"company": row["symbol"], "industry": metadata.get(row["symbol"], {}).get("industry") or "unknown",
                      "event_type": forecast["classification"], "horizon_minutes": str(forecast["horizon_minutes"]),
                      "conditions": forecast.get("conditions", "unknown")}
            for kind, label in labels.items():
                key = kind + ":" + label
                group = groups.setdefault(key, {"group": kind, "label": label, "n": 0, "mae": 0., "directional_accuracy": 0., "brier": 0.})
                group["n"] += 1
                group["mae"] += outcome["absolute_error"]
                group["directional_accuracy"] += int(outcome["direction_correct"])
                group["brier"] += outcome["brier"]
        for group in groups.values():
            for metric in ("mae", "directional_accuracy", "brier"):
                group[metric] /= group["n"]
        result["accuracy"] = list(groups.values())
        result["execution_boundary"] = "Read-only intelligence; orders remain under the existing paper controller."
        return result
