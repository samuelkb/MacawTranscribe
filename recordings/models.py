import uuid

from django.db import models
from django.utils import timezone


class RecordingStatus(models.TextChoices):
    """
    Enum for recording statuses
    """
    UPLOADED = "uploaded", "Uploaded"
    NORMALIZED = "normalized", "Normalized"
    DIARIZED = "diarized", "Diarized"
    TRANSCRIBING = "transcribing", "Transcribing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class ChunkStatus(models.TextChoices):
    """
    Enum for chunk recording statuses
    """
    PENDING = "pending", "Pending"
    QUEUED = "queued", "Queued"
    PROCESSING = "processing", "Processing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"
    NEEDS_REVIEW = "needs_review", "Needs Review"


class Recording(models.Model):
    """
    Model for recordings
    """
    id =  models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_file_name = models.CharField(max_length=512)
    original_file_path = models.TextField()
    normalized_file_path = models.TextField(blank=True)
    duration_milliseconds = models.BigIntegerField()
    status = models.CharField(
        max_length=32,
        choices=RecordingStatus.choices,
        default=RecordingStatus.UPLOADED,
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """
        Metaclass for recordings
        """
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.original_file_name} ({self.status})"


class Chunk(models.Model):
    """
    Model for chunks
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recording = models.ForeignKey(Recording, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    start_time = models.BigIntegerField()
    end_time = models.BigIntegerField()
    status = models.CharField(
        max_length=32,
        choices=ChunkStatus.choices,
        default=ChunkStatus.PENDING,
    )
    worker_id = models.CharField(max_length=128, blank=True, null=True)
    attempt_count = models.PositiveIntegerField(default=0)
    processing_started_at = models.DateTimeField(blank=True, null=True)
    heartbeat_at = models.DateTimeField(blank=True, null=True)
    last_error = models.TextField(blank=True)
    last_failed_at = models.DateTimeField(blank=True, null=True)
    retry_requested_model = models.CharField(max_length=64, blank=True, null=True)
    has_pending_candidate = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """
        Metaclass for chunks
        """
        ordering = ["-created_at", "chunk_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["recording", "chunk_index"],
                name="unique_chunk_index_per_recording"
            ),
            models.CheckConstraint(
                condition=models.Q(end_time__gt=models.F("start_time")),
                name="chunk_end_gt_start"
            )
        ]
        indexes = [
            models.Index(fields=["recording", "chunk_index"], name="chunk_rec_idx_idx"),
            models.Index(fields=["status"], name="chunk_status_idx"),
            models.Index(fields=["worker_id"], name="chunk_worker_idx"),
            models.Index(fields=["heartbeat_at"], name="chunk_heartbeat_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.recording_id}#{self.chunk_index} [{self.status}]"

    @property
    def duration_milliseconds(self) -> int:
        return self.end_time - self.start_time
