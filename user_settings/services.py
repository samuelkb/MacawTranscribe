import logging
from typing import Final
from uuid import UUID

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from ml.types import BackendName, ModelName
from user_settings.models import TranscriptionRuntimeSettings, WorkerProcessState, WorkerRole, WorkerStatus

logger: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_RUNTIME_SETTINGS: dict[str, object] = {
            "singleton_enforcer": True,
            "supervisor_enabled": True,
            "desired_worker_count": 2,
            "max_worker_count": 4,
            "max_job_per_worker": 3,
            "max_idle_seconds": 300,
            "supervisor_poll_seconds": 5,
            "guarded_scaling_enabled": True,
            "max_memory_percent": 85,
            "min_available_memory_mb": 12000,
            "estimated_memory_per_worker_mb": 12000,
            "cpu_guard_enabled": False,
            "max_cpu_percent": 85,
            "default_backend": BackendName.MLX_WHISPER.value,
            "default_model": ModelName.MEDIUM.value,
}

@transaction.atomic
def get_runtime_settings() -> TranscriptionRuntimeSettings:
    """
    Return the singleton runtime settings row.
    Recreate it if was manually deleted.
    """
    settings_object, created = TranscriptionRuntimeSettings.objects.select_for_update().get_or_create(
        pk=1,
        defaults=DEFAULT_RUNTIME_SETTINGS
    )
    if created:
        logger.warning(
            "transcription_runtime_settings_auto_created",
            extra={
                "settings_id": settings_object.pk,
            }
        )
    return settings_object

@transaction.atomic
def update_runtime_settings(**validated_fields: object) -> TranscriptionRuntimeSettings:
    """
    Update the singleton row using already-validated form data.
    """
    settings_object = get_runtime_settings()
    for field_name, value in validated_fields.items():
        setattr(settings_object, field_name, value)
    settings_object.save()
    logger.info(
        "transcription_runtime_settings_updated",
        extra={
            "settings_id": settings_object.pk,
            "updated_fields": sorted(validated_fields.keys()),
        }
    )
    return settings_object

def register_worker_process(*, worker_id: str, pid: int, role: WorkerRole, backend: BackendName | None = None, model: ModelName | None = None, hostname: str = "") -> WorkerProcessState:
    """
    Register the worker in Database.
    :param worker_id: Worker identifier
    :param pid: Worker process identifier
    :param role: Worker role, whether diarization or transcription
    :param backend: Worker backend
    :param model: Worker model
    :param hostname: Worker hostname machine
    """
    logger.info(
        "register_worker_process_started",
        extra={
            "worker_id": worker_id,
            "pid": pid,
            "role": role,
            "backend": backend,
            "model": model,
            "hostname": hostname,
        }
    )
    return WorkerProcessState.objects.update_or_create(
        worker_id=worker_id,
        defaults={
            "pid": pid,
            "role": role,
            "status": WorkerStatus.STARTING,
            "backend": backend.value if backend else None,
            "model": model.value if model else None,
            "hostname": hostname,
            "current_chunk_id": None,
            "idle_since": None,
            "jobs_processed": 0,
            "exit_reason": "",
            "last_error": "",
            "started_at": timezone.now(),
            "last_heartbeat_at": timezone.now(),
            "stopped_at": None,
        }
    )[0]

def heartbeat_worker(*, worker_id: str) -> None:
    """
    Updates the heartbeat state of the worker.
    :param worker_id: Worker identifier to update
    """
    logger.info(
        "heartbeat_worker_updating",
        extra={ "worker_id": worker_id, "heartbeat_at": timezone.now() },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        last_heartbeat_at=timezone.now()
    )

def mark_worker_idle(*, worker_id: str) -> None:
    """
    Updates the worker status to idle state.
    :param worker_id: Worker identifier to update
    """
    logger.info(
        "mark_worker_idle_updating",
        extra={ "worker_id": worker_id, "status": str(WorkerStatus.IDLE) },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        status=WorkerStatus.IDLE,
        current_chunk_id=None,
        idle_since=timezone.now(),
        last_heartbeat_at=timezone.now(),
    )

def mark_worker_busy(*, worker_id: str, chunk_id: UUID) -> None:
    """
    Updates the worker status to busy state.
    :param worker_id: Worker identifier to update
    :param chunk_id: Chunk identifier to assign to worker
    """
    logger.info(
        "mark_worker_busy_updating",
        extra={ "worker_id": worker_id, "status": str(WorkerStatus.BUSY) },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        status=WorkerStatus.BUSY,
        current_chunk_id=chunk_id,
        idle_since=None,
        last_heartbeat_at=timezone.now(),
    )

def increment_jobs_processed(*, worker_id: str) -> None:
    """
    Increment by one unit the jobs processed by the worker.
    :param worker_id: Worker identifier to increment
    """
    logger.info(
        "increment_jobs_processed_updating",
        extra={ "worker_id": worker_id },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        jobs_processed=F("jobs_processed") + 1,
        last_heartbeat_at=timezone.now(),
    )

def mark_worker_stopping(*, worker_id: str, exit_reason: str = "") -> None:
    """
    Updates the worker status to stopping state.
    :param worker_id: Worker identifier to update
    :param exit_reason: Exit reason string to mark the worker
    """
    logger.info(
        "mark_worker_stopping_updating",
        extra={ "worker_id": worker_id, "exit_reason": exit_reason },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        status=WorkerStatus.STOPPING,
        exit_reason=exit_reason,
        last_heartbeat_at=timezone.now(),
    )

def mark_worker_stopped(*, worker_id: str, exit_reason: str = "") -> None:
    """
    Updates the worker status to stopped state.
    :param worker_id: Worker identifier to update
    :param exit_reason: Exit reason string to mark the worker
    """
    logger.info(
        "mark_worker_stopped_updating",
        extra={ "worker_id": worker_id, "exit_reason": exit_reason },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        status=WorkerStatus.STOPPED,
        current_chunk_id=None,
        exit_reason=exit_reason,
        stopped_at=timezone.now(),
        last_heartbeat_at=timezone.now(),
    )

def mark_worker_failed(*, worker_id: str, error_message: str = "") -> None:
    """
    Updates the worker status to failed state.
    :param worker_id: Worker identifier to update
    :param error_message: Error message to mark the worker
    """
    logger.info(
        "mark_worker_failed_updating",
        extra={ "worker_id": worker_id, "error_message": error_message },
    )
    WorkerProcessState.objects.filter(worker_id=worker_id).update(
        status=WorkerStatus.FAILED,
        last_error=error_message,
        stopped_at=timezone.now(),
        last_heartbeat_at=timezone.now(),
    )

def get_default_worker_backend_and_model() -> tuple[BackendName | None, ModelName | None]:
    settings = TranscriptionRuntimeSettings.get_solo()
    backend = BackendName(settings.default_backend) if settings.default_backend else None
    model = ModelName(settings.default_model) if settings.default_model else None
    return backend, model