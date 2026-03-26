import logging
from typing import Final

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import CheckConstraint, Q
from django.utils import timezone

from ml.types import BackendName, ModelName

logger: Final[logging.Logger] = logging.getLogger(__name__)

class TranscriptionRuntimeSettings(models.Model):
    """
    Singleton model storing runtime control for worker supervision.
    """
    singleton_enforcer = models.BooleanField(default=True, editable=False, unique=True, help_text='Enforce a single RuntimeSettings row.')
    supervisor_enabled = models.BooleanField(default=True, help_text='Whether the worker supervisor should actively maintain workers.')
    desired_worker_count = models.PositiveIntegerField(default=2, help_text='Target number of transcription workers the supervisor should try to maintain.')
    max_worker_count = models.PositiveIntegerField(default=4, help_text='Hard cap for transcription workers, even if desired_worker_count is higher.')
    max_job_per_worker = models.PositiveIntegerField(default=3, help_text='Recycle a worker process after this many jobs.')
    max_idle_seconds = models.PositiveIntegerField(default=300, help_text='Recycle a worker if it stays idle for this many seconds.')
    supervisor_poll_seconds = models.PositiveIntegerField(default=5, help_text='How often the supervisor reconciles desired vs actual workers.')
    guarded_scaling_enabled = models.BooleanField(default=True, help_text='Whether the supervisor should block scale-up when machine resources are too tight.')
    max_memory_percent = models.PositiveIntegerField(default=85, help_text='Block worker scale-up when system memory usage is above this percentage.')
    min_available_memory_mb = models.PositiveIntegerField(default=12000, help_text='Minimum free memory required to allow spawning a new worker.')
    estimated_memory_per_worker_mb = models.PositiveIntegerField(default=12000, help_text='Estimated memory per worker.')
    cpu_guard_enabled = models.BooleanField(default=False, help_text="Whether CPU thresholds should participate in guarded scaling decisions.")
    max_cpu_percent = models.PositiveIntegerField(default=85, help_text="If cpu_guard_enabled is true, block scale-up when CPU usage is above this percentage.")
    default_backend = models.CharField(max_length=128, choices=[(b.value, b.name) for b in BackendName], blank=True, null=True)
    default_model = models.CharField(max_length=128, choices=[(m.value, m.name) for m in ModelName], blank=True, null=True)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Transcription Runtime Settings"
        verbose_name_plural = "Transcription Runtime Settings"

    def __str__(self) -> str:
        return "Transcription Runtime Settings"

    def clean(self) -> None:
        super().clean()

        if self.desired_worker_count > self.max_worker_count:
            raise ValidationError({
                "desired_worker_count": "desired_worker_count cannot be greater than max_worker_count.",
            })
        if self.max_worker_count == 0:
            raise ValidationError({
                "max_worker_count": "max_worker_count must be at least 1.",
            })
        if self.max_job_per_worker == 0:
            raise ValidationError({
                "max_job_per_worker": "max_job_per_worker must be at least 1.",
            })
        if self.supervisor_poll_seconds == 0:
            raise ValidationError({
                "supervisor_poll_seconds": "supervisor_poll_seconds must be at least 1.",
            })
        if self.cpu_guard_enabled and self.max_cpu_percent > 100:
            raise ValidationError({
                "max_cpu_percent": "max_cpu_percent cannot be greater than 100.",
            })
        if self.max_memory_percent > 100:
            raise ValidationError({
                "max_memory_percent": "max_memory_percent cannot be greater than 100.",
            })
        if self.max_memory_percent < 1:
            raise ValidationError({
                "max_memory_percent": "max_memory_percent must be at least 1.",
            })

    def save(self, *args, **kwargs):
        self.pk = 1
        self.full_clean()
        return super().save(*args, **kwargs)

    @classmethod
    def get_solo(cls) -> "TranscriptionRuntimeSettings":
        obj, _created = cls.objects.get_or_create(
            pk=1,
            defaults={
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
        )
        return obj


class WorkerRole(models.TextChoices):
    TRANSCRIPTION = "transcription", "Transcription"
    DIARIZATION = "diarization", "Diarization"


class WorkerStatus(models.TextChoices):
    STARTING = "starting", "Starting"
    IDLE = "idle", "Idle"
    BUSY = "busy", "Busy"
    STOPPING = "stopping", "Stopping"
    STOPPED = "stopped", "Stopped"
    FAILED = "failed", "Failed"


class WorkerProcessState(models.Model):
    worker_id = models.CharField(max_length=128, unique=True, help_text="Stable logical worker identifier assigned by the supervisor.")
    role = models.CharField(choices=WorkerRole.choices, default=WorkerRole.TRANSCRIPTION, help_text="Worker role handled by this process.")
    pid = models.PositiveIntegerField(unique=True, help_text="Operating system process ID.")
    status = models.CharField(choices=WorkerStatus.choices, default=WorkerStatus.STARTING, help_text="Current lifecycle status of the worker process.")
    backend = models.CharField(choices=[(b.value, b.name) for b in BackendName], blank=True, null=True, help_text="Backend loaded by this worker, when applicable.")
    model = models.CharField(choices=[(m.value, m.name) for m in ModelName], blank=True, null=True, help_text="Model loaded by this worker, when applicable.")
    current_chunk_id = models.UUIDField(blank=True, null=True, help_text="Chunk currently being processed by this worker, if any.")
    jobs_processed = models.PositiveIntegerField(default=0, help_text="Number of jobs completed by this worker process.")
    exit_reason = models.CharField(max_length=255, blank=True, default="", help_text="Reason recorded when the worker exits of is recycled.")
    last_error = models.TextField(blank=True, default="", help_text="Most recent error observed for this worker.")
    hostname = models.CharField(max_length=255, blank=True, default="", help_text="Host machine name running this worker.")
    started_at = models.DateTimeField(default=timezone.now, help_text="Date and time when this worker started.")
    last_heartbeat_at = models.DateTimeField(default=timezone.now, help_text="Date and time last heartbeat reported by this worker.")
    stopped_at = models.DateTimeField(blank=True, null=True, help_text="Date and time when this worker stopped, if applicable.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Worker Process State"
        verbose_name_plural = "Worker Process States"

        constraints = [
            CheckConstraint(
                condition=(
                        Q(status=WorkerStatus.BUSY, current_chunk_id__isnull=False) |
                        Q(~Q(status=WorkerStatus.BUSY), current_chunk_id__isnull=True)
                ),
                name="busy_chunk_bidirectional"
            )
        ]

    def __str__(self) -> str:
        return f"{self.worker_id} ({self.status})"
