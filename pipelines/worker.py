import logging
import socket
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from os import getpid
from typing import Final
from uuid import uuid4, UUID

from django.utils import timezone

from pipelines.queue import dequeue_transcription_job, QueueError
from pipelines.queue_types import TranscriptionJob
from recordings.models import Chunk, ChunkStatus
from transcriptions.services import transcribe_chunk, ChunkTranscriptionError

logger: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    """
    Runtime configuration for the transcription worker.
    """
    heartbeat_interval_seconds: int = 5
    stale_after_seconds: int = 30
    dequeue_timeout_seconds: int = 2
    recover_stale_chunks_on_startup: bool = True

def generate_worker_id() -> str:
    """
    Generate a readable worker identifier.
    """
    hostname =  socket.gethostname()
    pid = getpid()
    suffix = uuid4().hex[:8]
    return f"{hostname}:{pid}:{suffix}"

def update_chunk_heartbeat(*, chunk_id: UUID, worker_id: str) -> None:
    """
    Update the heartbeat for a chunk if it is still actively processing.
    :param chunk_id: UUID of the chunk to update
    :param worker_id: Worker identifier.
    """
    Chunk.objects.filter(id=chunk_id, status=ChunkStatus.PROCESSING, worker_id=worker_id).update(
        heartbeat_at=timezone.now(),
        updated_at=timezone.now(),
    )

def recover_stale_processing_chunks(*, stale_after_seconds: int) -> int:
    """
    Mark stale processing chunks as failed.

    A chunk is considered stale if status is PROCESSING and heartbeat is missing or too old.
    :param stale_after_seconds: Number of seconds before stale processing chunk was marked as stale.
    :return: Number of recovered chunks.
    """
    cutoff = timezone.now() - timedelta(seconds=stale_after_seconds)
    stale_chunks = Chunk.objects.filter(status=ChunkStatus.PROCESSING).filter(heartbeat_at__lt=cutoff)
    recovered_count = 0
    for chunk in stale_chunks:
        chunk.status = ChunkStatus.FAILED
        chunk.last_error = "worker heartbeat expired"
        chunk.last_failed_at = timezone.now()
        chunk.worker_id = ""
        chunk.save(update_fields=["status","last_error", "last_failed_at", "worker_id", "updated_at"])
        recovered_count += 1

    if recovered_count:
        logger.warning(
            "worker_recovered_stale_processing_chunks",
            extra={
                "recovered_count": recovered_count,
                "stale_after_seconds": stale_after_seconds,
            }
        )

    return recovered_count


class ChunkHeartbeatThread(threading.Thread):
    """
    Background heartbeat updater for one chunk while it is being processed.
    """

    def __init__(self, *, chunk_id: UUID, worker_id: str, heartbeat_interval_seconds: int) -> None:
        super().__init__(daemon=True)
        self.chunk_id = chunk_id
        self.worker_id = worker_id
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                update_chunk_heartbeat(chunk_id=self.chunk_id, worker_id=self.worker_id)
            except Exception:
                logger.exception(
                    "worker_heartbeat_update_failed",
                    extra={
                        "chunk_id": str(self.chunk_id),
                        "worker_id": str(self.worker_id),
                    }
                )
            self._stop_event.wait(self.heartbeat_interval_seconds)

def process_transcription_job(*, job: TranscriptionJob, worker_id: str, config: WorkerConfig) -> None:
    """
    Process one queued transcription job.
    :param job: Transcription job
    :param worker_id: Worker identifier.
    :param config: Worker configuration.
    """
    logger.info(
        "worker_job_processing_started",
        extra={
            "chunk_id": str(job.chunk_id),
            "backend": job.backend.value,
            "model": job.model.value,
            "worker_id": worker_id,
        }
    )
    heartbeat_thread = ChunkHeartbeatThread(
        chunk_id=job.chunk_id, worker_id=worker_id, heartbeat_interval_seconds=config.heartbeat_interval_seconds
    )
    heartbeat_thread.start()

    try:
        transcribe_chunk(
            chunk_id=job.chunk_id,
            backend=job.backend,
            model=job.model,
            worker_id=worker_id,
        )
    finally:
        heartbeat_thread.stop()
        heartbeat_thread.join(timeout=2)

    logger.info(
        "worker_job_processing_completed",
        extra={
            "chunk_id": str(job.chunk_id),
            "backend": job.backend.value,
            "model": job.model.value,
            "worker_id": worker_id,
        }
    )

def run_worker_loop(
        *,
        config: WorkerConfig | None = None,
        stop_event: threading.Event | None = None,
) -> None:
    """
    Run the transcription worker loop.

    - optionally recovers stale processing chunks on startup
    - continuously dequeues jobs from Redis
    - executes `transcribe_chunk(...)`
    - maintains heartbeat updates while processing
    :param config: Worker configuration.
    :param stop_event: Event to stop heartbeat processing.
    """
    config = config or WorkerConfig()
    stop_event = stop_event or threading.Event()
    worker_id = generate_worker_id()

    logger.info(
        "worker_loop_processing_started",
        extra={
            "worker_id": worker_id,
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "stale_after_seconds": config.stale_after_seconds,
            "dequeue_timeout_seconds": config.dequeue_timeout_seconds,
        }
    )

    if config.recover_stale_chunks_on_startup:
        recover_stale_processing_chunks(stale_after_seconds=config.stale_after_seconds)

    while not stop_event.is_set():
        try:
            job = dequeue_transcription_job(timeout_seconds=config.dequeue_timeout_seconds)
        except QueueError:
            logger.exception(
                "worker_queue_dequeue_failed",
                extra={"worker_id": worker_id}
            )
            time.sleep(1)
            continue

        if job is None:
            continue

        try:
            process_transcription_job(
                job=job,
                worker_id=worker_id,
                config=config,
            )
        except ChunkTranscriptionError:
            logger.warning(
                "worker_job_processing_failed",
                extra={
                    "chunk_id": str(job.chunk_id),
                    "backend": job.backend.value,
                    "model": job.model.value,
                    "worker_id": worker_id,
                }
            )
        except Exception:
            logger.exception(
                "worker_job_processing_unexpected_failure",
                extra={
                    "chunk_id": str(job.chunk_id),
                    "backend": job.backend.value,
                    "model": job.model.value,
                    "worker_id": worker_id,
                }
            )
    logger.info(
        "worker_loop_stopped",
        extra={"worker_id": worker_id},
    )