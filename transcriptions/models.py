import uuid

from django.db import models
from django.utils import timezone

from recordings.models import Chunk


class TranscriptWord(models.Model):
    """
    Word-level transcript output for a chunk.

    These rows are machine-generated and used for timing, alignment, playback synchronization and export features.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="transcript_words")
    word_index = models.PositiveIntegerField()
    text =  models.TextField()
    start_time = models.BigIntegerField()
    end_time = models.BigIntegerField()
    confidence = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["chunk_id", "word_index"]
        constraints = [
            models.UniqueConstraint(
                fields=["chunk_id", "word_index"],
                name="unique_word_index_per_chunk",
            ),
            models.CheckConstraint(
                condition=models.Q(end_time__gt=models.F("start_time")),
                name="transcript_word_end_gt_start"
            )
        ]
        indexes = [
            models.Index(
                fields=["chunk", "word_index"],
                name="transcript_word_chunk_idx"
            ),
            models.Index(
                fields=["chunk", "start_time"],
                name="transcript_word_start_idx"
            ),
        ]

    def __str__(self) -> str:
        return f"{self.chunk_id}#{self.word_index}: {self.text}"

    @property
    def duration_milliseconds(self) -> int:
        return self.end_time - self.start_time


class Transcript(models.Model):
    """
    Current accepted transcript for a chunk.

    This is the user-facing text shown in the editor. It may originate from the initial machine output, a later retry
    candidate or a human edit.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chunk = models.OneToOneField(Chunk, on_delete=models.CASCADE, related_name="transcript")
    accepted_text = models.TextField()
    model_used = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["chunk_id"]
        indexes = [
            models.Index(
                fields=["chunk_id"],
                name="transcript_chunk_idx"
            )
        ]

    def __str__(self) -> str:
        return f"{self.chunk_id}:accepted"


class Edit(models.Model):
    """
    Human edit history for a transcript.

    Every time the accepted transcript is changed by a human action, a new edit row is appended so previous versions can
    be restored later.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    transcript = models.ForeignKey(Transcript, on_delete=models.CASCADE, related_name="edits")
    edited_text = models.TextField()
    editor = models.CharField(max_length=255, default="user")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["transcript_id", "created_at"]
        indexes = [
            models.Index(
                fields=["transcript_id", "created_at"],
                name="edit_transcript_created_idx",
            )
        ]

    def __str__(self) -> str:
        return f"{self.transcript_id}:edit@{self.created_at.isoformat()}"


class TranscriptCandidate(models.Model):
    """
    Machine-generated alternative transcript for a chunk.

    This is typically created when a chunk is retried with another model after the transcript already exists and may
    already have human edits.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="transcript_candidates")
    candidate_text = models.TextField()
    model_used = models.CharField(max_length=255)
    confidence = models.FloatField(null=True, blank=True)
    is_from_retry = models.BooleanField(default=True)
    accepted = models.BooleanField(default=False)
    rejected = models.BooleanField(default=False)
    accepted_at = models.DateTimeField(null=True, blank=True)
    rejected_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["chunk_id", "-created_at"]
        indexes = [
            models.Index(
                fields=["chunk_id", "created_at"],
                name="candidate_chunk_created_idx"
            ),
            models.Index(
                fields=["chunk_id", "accepted"],
                name="candidate_chunk_accepted_idx"
            )
        ]

    def __str__(self) -> str:
        return f"{self.chunk_id}:candidate:{self.model_used}"
