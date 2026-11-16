"""Import the backend application when tests run from the repository root."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.fixture(autouse=True)
def local_model_defaults(monkeypatch):
    """Keep contract tests independent of the owner's installed model settings."""
    import os

    from app.config import Settings, settings

    for key in list(os.environ):
        if key.startswith("OLLAMA_"):
            monkeypatch.delenv(key)
    for key, field in Settings.model_fields.items():
        if key.startswith("OLLAMA_"):
            monkeypatch.setattr(settings, key, field.default)
