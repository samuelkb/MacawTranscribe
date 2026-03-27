import json
import logging
from typing import Final

from django.conf import settings
from redis import Redis

from ml.types import BackendName, ModelName
from pipelines.queue_names import build_transcription_queue_name, build_transcription_partition_key
from pipelines.queue_types import TranscriptionJob

logger: Final[logging.Logger] = logging.getLogger(__name__)


class QueueError(RuntimeError):
    """Raised when queue operations fail."""

"""
def get_queue_name() -> str:
    """"""Get queue name.""""""
    logger.debug("Running get_queue_name")
    return getattr(settings, "TRANSCRIPTION_QUEUE_NAME", "transcription_jobs")
"""
def get_redis_client() -> Redis:
    logger.debug("redis_client_initializing")

    client = Redis(
        host=getattr(settings, "REDIS_HOST", "127.0.0.1"),
        port=getattr(settings, "REDIS_PORT", 6379),
        db=getattr(settings, "REDIS_DB", 0),
        password=getattr(settings, "REDIS_PASSWORD", None),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )

    try:
        client.ping()
        logger.debug("redis_client_connected")
    except Exception as exc:
        logger.exception("redis_client_connection_failed")
        raise

    return client

def enqueue_transcription_job(*, job: TranscriptionJob) -> None:
    """
    Push one transcription job to Redis queue.
    :param job: transcription job
    """
    logger.debug("Running enqueue_transcription_job")
    client = get_redis_client()
    queue_name = build_transcription_queue_name(backend=job.backend, model=job.model)
    partition_key = build_transcription_partition_key(backend=job.backend, model=job.model)
    try:
        client.rpush(queue_name, json.dumps(job.to_dict()))
    except Exception as exc:
        raise QueueError("Failed to enqueue transcription job") from exc

    logger.info(
        "transcription_job_enqueued",
        extra={
            "queue": queue_name,
            "chunk_id": str(job.chunk_id),
            "backend": job.backend.value,
            "model": job.model.value,
            "partition_key": partition_key,
            "queue_name": queue_name,
        }
    )

def dequeue_transcription_job(*, backend: BackendName, model: ModelName, timeout_seconds: int = 1) -> TranscriptionJob | None:
    """
    Pop one transcription job from the Redis queue.
    :param timeout_seconds: timeout in seconds
    :param backend: Transcription backend identifier
    :param model: Transcription model identifier
    :return: A job if available, otherwise None on timeout
    """
    logger.debug("Running dequeue_transcription_job")
    client = get_redis_client()
    queue_name = build_transcription_queue_name(backend=backend, model=model)
    partition_key = build_transcription_partition_key(backend=backend, model=model)
    try:
        result = client.blpop(queue_name, timeout=timeout_seconds)
    except Exception as exc:
        raise QueueError("Failed to dequeue transcription job") from exc
    if result is None:
        return None

    _, raw_payload = result

    try:
        payload = json.loads(raw_payload)
        job: TranscriptionJob = TranscriptionJob.from_dict(payload)
    except Exception as exc:
        raise QueueError("invalid transcription job payload in queue") from exc
    logger.info(
        "transcription_job_enqueued",
        extra={
            "queue": queue_name,
            "chunk_id": str(job.chunk_id),
            "backend": job.backend.value,
            "model": job.model.value,
            "partition_key": partition_key,
            "queue_name": queue_name,
        }
    )
    return job
