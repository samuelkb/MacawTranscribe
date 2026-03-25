import json
import logging
from typing import Final

from django.conf import settings
from redis import Redis

from pipelines.queue_types import TranscriptionJob

logger: Final[logging.Logger] = logging.getLogger(__name__)


class QueueError(RuntimeError):
    """Raised when queue operations fail."""


def get_queue_name() -> str:
    """Get queue name."""
    logger.debug("Running get_queue_name")
    return getattr(settings, "TRANSCRIPTION_QUEUE_NAME", "transcription_jobs")

def get_redis_client() -> Redis:
    """Get Redis client."""
    logger.debug("Running get_redis_client")
    redis_url: str = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(redis_url, decode_responses=True)

def enqueue_transcription_job(*, job: TranscriptionJob) -> None:
    """
    Push one transcription job to Redis queue.
    :param job: transcription job
    """
    logger.debug("Running enqueue_transcription_job")
    client = get_redis_client()
    queue_name = get_queue_name()
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
        }
    )

def dequeue_transcription_job(*, timeout_seconds: int = 1) -> TranscriptionJob | None:
    """
    Pop one transcription job from the Redis queue.
    :param timeout_seconds: timeout in seconds
    :return: A job if available, otherwise None on timeout
    """
    logger.debug("Running dequeue_transcription_job")
    client = get_redis_client()
    queue_name = get_queue_name()
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
        }
    )
    return job
