import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final
from uuid import UUID

from django.db import transaction, close_old_connections
from django.utils import timezone

from ml.types import BackendName, ModelName
from pipelines.chunk_heartbeat import update_chunk_heartbeat
from pipelines.events import publish_workspace_event
from pipelines.services import get_recording_process
from recordings.audio import extract_chunk_audio
from recordings.models import Chunk, ChunkStatus, RecordingStatus
from transcriptions.models import TranscriptWord, Transcript, TranscriptCandidate, Edit
from transcriptions.runtime import LoadedWorkerRuntime
from transcriptions.services_runtime import load_worker_transcription_runtime
from user_settings.services import heartbeat_worker

logger: Final[logging.Logger] = logging.getLogger(__name__)


class ChunkTranscriptionError(RuntimeError):
    """Raised when chunk transcription fails."""


def _publish_chunk_updated(*, chunk: Chunk, accepted_text: str = "", message: str = "") -> None:
    """
    Publish the current persisted state for one chunk to the workspace event stream.
    :param chunk: Chunk whose current state should be emitted.
    :param accepted_text: Accepted transcript text when available.
    :param message: Optional error or status message for the UI.
    :return: None
    """
    publish_workspace_event(
        recording_id=chunk.recording_id,
        event_type="chunk_updated",
        payload={
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "status": chunk.status,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "has_transcript": bool(accepted_text.strip()),
            "accepted_text": accepted_text.strip(),
            "message": message,
        },
    )


def _publish_chunk_progress(*, recording_id: UUID) -> None:
    """
    Publish the current aggregate chunk progress counters for one recording.
    :param recording_id: Recording identifier.
    :return: None
    """
    publish_workspace_event(
        recording_id=recording_id,
        event_type="chunk_progress",
        payload=get_recording_process(recording_id=recording_id),
    )


def _build_transcription_heartbeat_callback(
        *,
        chunk_id: UUID,
        worker_id: str | None,
        min_interval_seconds: float = 5.0
) -> Callable[[], None]:
    last_sent_at = 0.0
    def heartbeat() -> None:
        nonlocal last_sent_at
        now = time.monotonic()
        if now - last_sent_at < min_interval_seconds:
            return
        last_sent_at = now
        close_old_connections()
        try:
            if worker_id:
                heartbeat_worker(worker_id=worker_id)
            update_chunk_heartbeat(chunk_id=chunk_id, worker_id=worker_id)
        finally:
            close_old_connections()
    return heartbeat

def persist_transcription_words(*, chunk: Chunk, words: tuple, model_used: str) -> list[TranscriptWord]:
    """
    Replace the machine-generated word-level transcript rows for a chunk.

    Existing TranscriptWord rows for the chunk are deleted and replaced.
    :param chunk: Chunk whose word-level transcription should be replaced.
    :param words: Sequence of backend word objects compatible with the ML backend contract.
    :param model_used: Model used to transcription.
    :return: Persisted transcript word rows.
    """
    logger.info(
        "persist_transcription_words_started",
        extra={
            "chunk": chunk,
            "num_words": len(words)
        },
    )
    TranscriptWord.objects.filter(chunk=chunk).delete()
    logger.info(f"pre-existing words for chunk {chunk.id} deleted")
    created_words: list[TranscriptWord] = TranscriptWord.objects.bulk_create(
        [
            TranscriptWord(
                chunk=chunk,
                word_index=word.word_index,
                text=word.text,
                start_time=word.start_time,
                end_time=word.end_time,
                confidence=word.confidence,
                model_used=model_used,
            )
            for word in words
        ]
    )
    logger.info(
        "persist_transcription_words_completed",
        extra={
            "chunk": chunk,
            "num_words": len(words)
        }
    )
    return list(created_words)

def create_initial_transcript(*, chunk: Chunk, accepted_text: str, model_used: str | None) -> Transcript:
    """
    Create the first accepted transcript for a chunk.
    :param chunk: Chunk receiving its initial accepted transcript.
    :param accepted_text: Initial machine-generated transcript text.
    :param model_used: Model used to create the text.
    :return: Newly created Transcript row.
    :raises ValueError: If chunk already has an accepted transcript.
    """
    logger.info(
        "create_initial_transcript_started",
        extra={
            "chunk": chunk,
            "accepted_text": accepted_text,
            "model_used": model_used,
        }
    )
    if Transcript.objects.filter(chunk=chunk).exists():
        raise ValueError("chunk already has an accepted transcript")
    return Transcript.objects.create(
        chunk=chunk,
        accepted_text=accepted_text,
        model_used=model_used,
    )

def create_transcript_candidate(
        *,
        chunk: Chunk,
        candidate_text: str,
        model_used: str | None,
        confidence: float | None = None,
        is_from_retry: bool = True,
)->TranscriptCandidate:
    """
    Create a machine-generated transcript candidate for a chunk.
    :param chunk: Chunk receiving the candidate transcript.
    :param candidate_text: Candidate transcript text.
    :param model_used: Model used to create the candidate text.
    :param confidence: Optional Confidence score for candidate transcript.
    :param is_from_retry: Whether the candidate transcript came from a retry workflow
    :return: Candidate transcript candidate row.
    """
    logger.info(
        "create_transcript_candidate_started",
        extra={
            "chunk": chunk,
            "candidate_text": candidate_text,
            "model_used": model_used,
            "confidence": confidence,
            "is_from_retry": is_from_retry,
        }
    )
    candidate: TranscriptCandidate = TranscriptCandidate.objects.create(
        chunk=chunk,
        candidate_text=candidate_text,
        model_used=model_used,
        confidence=confidence,
        is_from_retry=is_from_retry,
    )
    if not chunk.has_pending_candidate:
        chunk.has_pending_candidate = True
        chunk.save(update_fields=["has_pending_candidate", "updated_at"])
    logger.info(
        "create_transcript_candidate_completed",
        extra={
            "chunk": chunk,
            "candidate_text": candidate_text,
            "model_used": model_used,
            "confidence": confidence,
            "is_from_retry": is_from_retry,
        }
    )
    return candidate

def apply_candidate(*, candidate: TranscriptCandidate) -> Transcript:
    """
    Apply a transcript candidate as the current accepted transcript.

    - Updates or creates the chunk's accepted transcript.
    - Marks the selected candidate as accepted.
    - Marks other non-finalized candidates for the same chunk as rejected.
    - Clears chunk.has_pending_candidate.
    :param candidate: Candidate transcript candidate row.
    :return: Updated accepted transcript row.
    """
    logger.info(
        "apply_candidate_started",
        extra={
            "candidate": candidate.__str__(),
        }
    )
    now = timezone.now()
    chunk: Chunk = candidate.chunk
    with transaction.atomic():
        transcript, _ = Transcript.objects.get_or_create(
            chunk=chunk,
            defaults={
                "accepted_text": candidate.candidate_text,
                "model_used": candidate.model_used,
            }
        )
        transcript.accepted_text = candidate.candidate_text
        transcript.model_used = candidate.model_used
        transcript.save(update_fields=["accepted_text", "model_used", "updated_at"])
        logger.info(
            "save_transcript_update",
            extra={
                "transcript": transcript.__str__(),
            }
        )
        candidate.accepted = True
        candidate.rejected = False
        candidate.accepted_at = now
        candidate.rejected_at =None
        candidate.save(update_fields=["accepted", "rejected", "accepted_at", "rejected_at"])
        logger.info(
            "save_candidate_update",
            extra={
                "candidate": candidate.__str__(),
            }
        )
        TranscriptCandidate.objects.filter(chunk=chunk).exclude(id=candidate.id).filter(
            accepted=False,
            rejected=False,
        ).update(
            rejected=True,
            rejected_at=now,
        )
        logger.info("mark_other_non_finalized_candidates_rejected")
        chunk.has_pending_candidate = False
        chunk.save(update_fields=["has_pending_candidate", "updated_at"])

    return transcript

def append_edit(*, transcript: Transcript, edited_text: str, editor: str = "user") -> Edit:
    """
    Append a human edit and update the current accepted transcript.
    :param transcript: Transcript being edited.
    :param edited_text: New accepted text.
    :param editor: Identifier for the editor.
    :return: Persisted edit row.
    """
    logger.info(
        "append_edit_started",
        extra={
            "transcript": transcript.__str__(),
            "edited_text": edited_text,
            "editor": editor,
        }
    )
    with transaction.atomic():
        edit = Edit.objects.create(
            transcript=transcript,
            edited_text=edited_text,
            editor=editor,
        )
        transcript.accepted_text = edited_text
        transcript.save(update_fields=["accepted_text", "updated_at"])
    logger.info(
        "append_edit_completed",
        extra={
            "transcript": transcript.__str__(),
            "edited_text": edited_text,
            "editor": editor,
        }
    )
    return edit

def _mark_chunk_processing(*, chunk: Chunk, worker_id: str | None) -> None:
    logger.info(
        "_mark_chunk_processing_started",
        extra={
            "chunk": chunk.__str__(),
            "worker_id": worker_id,
        }
    )
    chunk.status = ChunkStatus.PROCESSING
    chunk.attempt_count += 1
    chunk.processing_started_at = timezone.now()
    chunk.heartbeat_at = timezone.now()
    chunk.worker_id = worker_id
    chunk.save(update_fields=[
        "status",
        "attempt_count",
        "processing_started_at",
        "heartbeat_at",
        "worker_id",
        "updated_at",
    ])
    logger.info(
        "_mark_chunk_processing_completed",
        extra={
            "chunk": chunk.__str__(),
            "worker_id": worker_id,
            "updated_at": chunk.updated_at,
        }
    )

def _mark_chunk_completed(*, chunk: Chunk) -> None:
    logger.info(
        "_mark_chunk_completed_started",
        extra={
            "chunk": chunk.__str__(),
        }
    )
    chunk.status = ChunkStatus.COMPLETED
    chunk.last_error = ""
    chunk.last_failed_at = None
    chunk.heartbeat_at = timezone.now()
    chunk.save(update_fields=[
        "status",
        "last_error",
        "last_failed_at",
        "heartbeat_at",
        "updated_at",
    ])
    logger.info(
        "_mark_chunk_completed_completed",
        extra={
            "chunk": chunk.__str__(),
            "updated_at": chunk.updated_at,
        }
    )

def _mark_chunk_failed(*, chunk: Chunk, error_message: str) -> None:
    logger.info(
        "_mark_chunk_failed_started",
        extra={
            "chunk": chunk.__str__(),
            "error_message": error_message,
        }
    )
    chunk.status = ChunkStatus.FAILED
    chunk.last_error = error_message
    chunk.last_failed_at = timezone.now()
    chunk.heartbeat_at = timezone.now()
    chunk.save(update_fields=[
        "status",
        "last_error",
        "last_failed_at",
        "heartbeat_at",
        "updated_at",
    ])
    logger.info(
        "_mark_chunk_failed_completed",
        extra={
            "chunk": chunk.__str__(),
            "updated_at": chunk.updated_at,
        }
    )

def _derive_full_text_from_words(words: tuple) -> str:
    """
    Build transcript text from backend words.
    """
    logger.info("_derive_full_text_from_words_started")
    return "".join(words.text.strip() for word in words if word.text.strip()).strip()

def transcribe_chunk_with_runtime(
        *,
        chunk_id: UUID,
        runtime: LoadedWorkerRuntime,
        worker_id: str | None = None,
) -> Transcript:
    """
    Transcribe one chunk end-to-end.

    Flow:
    1. Load chunk and recording
    2. Mark chunk as processing
    3. Extract chunk audio on demand
    4. Transcribe chunk audio using the preloaded runtime
    5. Replace TranscriptWord rows
    6. Create initial Transcript or create a TranscriptCandidate
    7. Mark chunk completed
    8. Update recording completion status
    :param chunk_id: Chunk identifier.
    :param runtime: Preloaded worker transcription runtime.
    :param worker_id: Optional worker identifier for heartbeat/debugging.
    :return: The current accepted transcript for the chunk.
    :raises Chunk.DoesNotExist: Chunk does not exist.
    :raises ChunkTranscriptionError: If transcription fails.
    """
    chunk = (Chunk.objects.select_related("recording").get(id=chunk_id))
    if not chunk.recording.normalized_file_path or not chunk.recording.normalized_file_path.strip():
        raise ChunkTranscriptionError("recording.normalized_file_path must not be empty")
    logger.info(
        "chunk_transcription_started",
        extra={
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "recording_id": str(chunk.recording.id),
            "worker_id": worker_id,
            "backend": runtime.backend.value,
            "model": runtime.model.value,
            "partition_key": runtime.partition_key,
        }
    )

    _mark_chunk_processing(chunk=chunk, worker_id=worker_id)
    _publish_chunk_updated(chunk=chunk)
    _publish_chunk_progress(recording_id=chunk.recording_id)

    temp_audio_path: Path | None = None

    try:
        temp_audio_path = extract_chunk_audio(chunk=chunk)
        heartbeat_callback = _build_transcription_heartbeat_callback(chunk_id=chunk_id, worker_id=worker_id)
        result = runtime.backend_impl.transcribe(
            loaded_model=runtime.loaded_model,
            audio_path=temp_audio_path,
            heartbeat_callback=heartbeat_callback,
        )
        close_old_connections()
        model_used_value = result.model_used.value
        persisted_words = persist_transcription_words(chunk=chunk,words=result.words, model_used=model_used_value)
        accepted_text = result.full_text.strip() or _derive_full_text_from_words(result.words)

        try:
            transcript = create_initial_transcript(
                chunk=chunk,
                accepted_text=accepted_text,
                model_used=model_used_value,
            )
            created_candidate = False
        except ValueError:
            transcript = chunk.transcript
            create_transcript_candidate(
                chunk=chunk,
                candidate_text=accepted_text,
                model_used=model_used_value,
                confidence=None,
                is_from_retry=True,
            )
            created_candidate = True

        _mark_chunk_completed(chunk=chunk)
        update_recording_completion_status(chunk=chunk)
        chunk.refresh_from_db()
        _publish_chunk_updated(chunk=chunk, accepted_text=transcript.accepted_text)
        _publish_chunk_progress(recording_id=chunk.recording_id)

        logger.info(
            "chunk_transcription_completed",
            extra={
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "recording_id": str(chunk.recording_id),
                "worker_id": worker_id,
                "backend_used": runtime.backend.value,
                "model_used": runtime.model.value,
                "word_count": len(persisted_words),
                "created_candidate": created_candidate,
                "status": chunk.status,
            },
        )

        return transcript

    except Exception as exc:
        error_message = str(exc)
        _mark_chunk_failed(chunk=chunk, error_message=error_message)
        chunk.refresh_from_db()
        _publish_chunk_updated(chunk=chunk, message=error_message)
        _publish_chunk_progress(recording_id=chunk.recording_id)

        logger.exception(
            "chunk_transcription_failed",
            extra={
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "recording_id": str(chunk.recording_id),
                "worker_id": worker_id,
                "backend_used": runtime.backend.value,
                "model_used": runtime.model.value,
                "error": error_message,
            },
        )
        raise ChunkTranscriptionError(error_message) from exc

    finally:
        if temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)
        close_old_connections()

def transcribe_chunk_on_demand(
    *,
    chunk_id: UUID,
    backend: BackendName | None = None,
    model: ModelName | None = None,
    worker_id: str | None = None,
) -> Transcript:
    runtime = load_worker_transcription_runtime(
        backend=backend,
        model=model,
    )
    return transcribe_chunk_with_runtime(
        chunk_id=chunk_id,
        runtime=runtime,
        worker_id=worker_id,
    )

def update_recording_completion_status(*, chunk: Chunk) -> None:
    """
    Mark the parent recording as completed if all chunks are completed.
    :param chunk: Chunk identifier.
    """
    recording = chunk.recording
    has_incomplete_chunks = Chunk.objects.filter(recording=recording).exclude(
        status=ChunkStatus.COMPLETED
    ).exists()

    if not has_incomplete_chunks:
        recording.status = RecordingStatus.COMPLETED
        recording.save(update_fields=["status", "updated_at"])
