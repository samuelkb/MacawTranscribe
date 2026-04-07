import logging
import socket
import threading
import time
from dataclasses import dataclass
from os import getpid
from typing import Final
from uuid import uuid4

from django.db import close_old_connections

from pipelines.queue import dequeue_workspace_pipeline_job, QueueError
from pipelines.queue_names import build_workspace_pipeline_queue_name
from pipelines.queue_types import WorkspacePipelineJob
from pipelines.services import run_workspace_pipeline

logger: Final[logging.Logger] = logging.getLogger(__name__)


class WorkspacePipelineError(RuntimeError):
    """Raised when the workspace pipeline worker cannot process a job."""


@dataclass(frozen=True)
class WorkspaceWorkerConfig:
    """
    Runtime configuration for the workspace pipeline worker.
    """
    dequeue_timeout_seconds: int = 2
    idle_sleep_seconds: float = 0.5


def generate_workspace_worker_id() -> str:
    """
    Generate a readable workspace worker identifier.
    """
    hostname = socket.gethostname()
    pid = getpid()
    suffix = uuid4().hex[:8]
    return f"{hostname}:{pid}:{suffix}"


def process_workspace_pipeline_job(*, job: WorkspacePipelineJob, worker_id: str) -> None:
    """
    Process one workspace pipeline job.
    :param job: Workspace pipeline job
    :param worker_id: Worker identifier.
    """
    close_old_connections()
    logger.info(
        "workspace_worker_job_processing_started",
        extra={
            "recording_id": str(job.recording_id),
            "worker_id": worker_id,
        }
    )

    try:
        run_workspace_pipeline(recording_id=job.recording_id)
    except Exception as exc:
        logger.exception(
            "workspace_worker_job_processing_failed",
            extra={
                "recording_id": str(job.recording_id),
                "worker_id": worker_id,
                "error": str(exc),
            }
        )
        raise WorkspacePipelineError(str(exc)) from exc
    finally:
        close_old_connections()

    logger.info(
        "workspace_worker_job_processing_completed",
        extra={
            "recording_id": str(job.recording_id),
            "worker_id": worker_id,
        }
    )


def run_workspace_worker_loop(
        *,
        worker_id: str,
        config: WorkspaceWorkerConfig | None = None,
        stop_event=None
) -> None:
    """
    Run the workspace pipeline worker loop.

    - continuously dequeues workspace pipeline jobs from Redis
    - executes the recording preprocessing orchestration
    :param worker_id: Worker identifier.
    :param config: Worker configuration.
    :param stop_event: Event to stop worker loop.
    """
    config = config or WorkspaceWorkerConfig()
    stop_event = stop_event or threading.Event()
    queue_name = build_workspace_pipeline_queue_name()

    logger.info(
        "workspace_worker_loop_started",
        extra={
            "worker_id": worker_id,
            "dequeue_timeout_seconds": config.dequeue_timeout_seconds,
            "idle_sleep_seconds": config.idle_sleep_seconds,
            "queue_name": queue_name,
        }
    )

    try:
        while not stop_event.is_set():
            close_old_connections()
            try:
                job = dequeue_workspace_pipeline_job(timeout_seconds=config.dequeue_timeout_seconds)
            except QueueError:
                logger.exception(
                    "workspace_worker_queue_dequeue_failed",
                    extra={"worker_id": worker_id}
                )
                close_old_connections()
                time.sleep(1)
                continue

            if job is None:
                close_old_connections()
                time.sleep(config.idle_sleep_seconds)
                continue

            try:
                process_workspace_pipeline_job(
                    job=job,
                    worker_id=worker_id,
                )
            except WorkspacePipelineError:
                logger.warning(
                    "workspace_worker_job_failed",
                    extra={
                        "recording_id": str(job.recording_id),
                        "worker_id": worker_id,
                    }
                )
            except Exception:
                logger.exception(
                    "workspace_worker_job_unexpected_failure",
                    extra={
                        "recording_id": str(job.recording_id),
                        "worker_id": worker_id,
                    }
                )

        logger.info(
            "workspace_worker_loop_stopped",
            extra={"worker_id": worker_id},
        )
    finally:
        close_old_connections()
