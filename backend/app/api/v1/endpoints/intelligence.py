"""Read-only monitoring of the intelligence archive."""
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.intelligence.store import IntelligenceStore, sha, stamp
from app.intelligence.controller import IntelligenceController

router = APIRouter()


def archive():
    if not Path(settings.INTELLIGENCE_DB_PATH).exists():
        raise HTTPException(404, "Intelligence has not acquired evidence yet.")
    store = IntelligenceStore.__new__(IntelligenceStore)
    store.path, store.read_only = settings.INTELLIGENCE_DB_PATH, True
    return store


@router.get("/status")
def status():
    if not Path(settings.INTELLIGENCE_DB_PATH).exists():
        return {"status": "waiting", "counts": {}, "sources": [], "coverage": [],
                "investigations": [], "forecasts": [], "lessons": [], "activity": [], "accuracy": []}
    controller = IntelligenceController.__new__(IntelligenceController)
    controller.store = archive()
    return {"status": "active", **controller.overview()}


@router.get("/evidence/{chunk_id}")
def evidence(chunk_id: str, asof: str | None = None):
    store, now = archive(), stamp(asof)
    with store.connect() as db:
        row = db.execute("""SELECT c.*,v.available,v.published,v.acquired,v.event_time,d.url,d.publisher,d.title
            FROM document_chunks c JOIN document_versions v ON c.version=v.id
            JOIN documents d ON v.document=d.id WHERE c.id=? AND v.available<=?""", (chunk_id, now)).fetchone()
    if not row:
        raise HTTPException(404, "Evidence is unavailable at this decision time.")
    result = dict(row)
    if sha(result["body"]) != result["hash"]:
        raise HTTPException(409, "Evidence integrity check failed.")
    return result


@router.get("/graph")
def graph(asof: str | None = None, limit: int = Query(100, ge=1, le=500)):
    store, now = archive(), stamp(asof)
    with store.connect() as db:
        rows = db.execute("""SELECT r.*,a.name subject_name,b.name object_name FROM entity_relationships r
            JOIN entities a ON r.subject=a.id JOIN entities b ON r.object=b.id
            WHERE r.available<=? ORDER BY r.available DESC LIMIT ?""", (now, limit)).fetchall()
    return {"asof": now, "relationships": [dict(row) for row in rows]}


@router.get("/original/{version}")
def original(version: str, asof: str | None = None):
    store, now = archive(), stamp(asof)
    with store.connect() as db:
        row = db.execute("SELECT id,hash,original_text,published,acquired,available FROM document_versions WHERE id=? AND available<=?", (version, now)).fetchone()
    if not row:
        raise HTTPException(404, "Version is unavailable at this decision time.")
    result = dict(row)
    if sha(result["original_text"]) != result["hash"]:
        raise HTTPException(409, "Original evidence integrity check failed.")
    return result
