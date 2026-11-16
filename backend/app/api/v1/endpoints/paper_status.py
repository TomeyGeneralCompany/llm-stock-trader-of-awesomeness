"""Read-only status of the configured local paper account."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.paper_trader import read_report

router = APIRouter()


@router.get("/status")
def status():
    if not Path(settings.PAPER_DB_PATH).is_file():
        raise HTTPException(status_code=404, detail="Paper account has not been initialized.")
    try:
        return read_report(settings.PAPER_DB_PATH)
    except (sqlite3.Error, KeyError, TypeError, ValueError):
        raise HTTPException(status_code=503, detail="Paper account cannot be read right now.")
