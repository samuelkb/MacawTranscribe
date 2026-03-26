import logging
import os
import threading
from typing import Final

from django.conf import settings

from pipelines.worker import run_worker_loop, WorkerConfig

logger: Final[logging.Logger] = logging.getLogger(__name__)

_worker_threads_stated = False
_worker_threads: list[threading.Thread] = []
"""
def should_start_embedded_workers() -> bool:
    if not getattr(settings, "PIPELINES_START_EMBEDDED_WORKERS", True):
        return False
    run_main = os.environ.get("RUN_MAIN")
    if run_main not in {"true", "True",None}:
        return False
    return True

def start_embedded_workers() -> None:
    global _worker_threads_stated

    if _worker_threads_stated:
        return

    if not should_start_embedded_workers():
        logger.info(
            "embedded_workers_not_started"
        )
        return

    worker_count = getattr(settings, "PIPELINES_WORKER_COUNT", 1)
    logger.info(
        "embedded_workers_starting",
        extra={"worker_count": worker_count}
    )

    for index in range(worker_count):
        stop_event = threading.Event()
        thread = threading.Thread(
            target=run_worker_loop,
            kwargs={
                "config": WorkerConfig(),
                "stop_event": stop_event,
            },
            daemon=True,
            name=f"transcription-worker-{index + 1}"
        )
        thread.start()
        _worker_threads.append(thread)

    _worker_threads_stated = True
    logger.info("embedded_workers_started", extra={"worker_count": worker_count})
"""