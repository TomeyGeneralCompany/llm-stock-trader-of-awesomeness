"""Shared resource ownership for pipeline stages and local inference."""

import threading
from contextlib import contextmanager

from filelock import FileLock
from threadpoolctl import threadpool_limits

from app.config import settings

_work_slot = threading.RLock()
_owner = threading.local()
_file_slots = {}


@contextmanager
def _resource_slot():
    """Serialize threads and backend processes; OS locks release after a crash."""
    from pathlib import Path

    with _work_slot:
        path = settings.PIPELINE_LOCK_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if path not in _file_slots:
            _file_slots[path] = FileLock(path, timeout=-1)
        with _file_slots[path]:
            yield


class StageResourceError(RuntimeError):
    """An operation is not allowed in the active stage."""


def require_acquisition_stage():
    """Market-data downloads belong to ACQUIRE when a pipeline owns the thread."""
    stage = getattr(_owner, "stage", None)
    if stage is not None and stage != "ACQUIRE":
        raise StageResourceError(
            f"Market-data acquisition is not allowed during {stage}."
        )


@contextmanager
def stage_resources(stage: str):
    """Allow one heavy stage at a time and cap numerical library thread pools."""
    with _resource_slot(), threadpool_limits(limits=settings.PIPELINE_NUM_THREADS):
        previous = getattr(_owner, "stage", None)
        if previous is not None:
            raise StageResourceError("A pipeline stage cannot start another stage.")
        _owner.stage = stage
        try:
            yield
        finally:
            _owner.stage = previous


@contextmanager
def inference_resources():
    """Permit pipeline inference only during ANALYZE; serialize other AI tools."""
    with _resource_slot():
        stage = getattr(_owner, "stage", None)
        if stage is not None and stage != "ANALYZE":
            raise StageResourceError(f"Model inference is not allowed during {stage}.")
        yield
