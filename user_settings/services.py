import logging
from typing import Final

from django.db import transaction

from ml.types import BackendName, ModelName
from user_settings.models import TranscriptionRuntimeSettings

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
