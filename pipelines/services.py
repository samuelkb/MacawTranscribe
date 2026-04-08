import logging
from typing import Final
from dataclasses import dataclass
from uuid import UUID

from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.urls import reverse

from ml.manager import ModelManager, ResolvedModelSelection
from ml.types import BackendName, ModelName
from pipelines.events import publish_workspace_event
from pipelines.queue import enqueue_transcription_job, enqueue_workspace_pipeline_job
from pipelines.queue_types import TranscriptionJob, WorkspacePipelineJob
from recordings.models import Recording, RecordingStatus, ChunkStatus, Chunk
from recordings.services import ingest_uploaded_recording, normalize_audio, create_chunks, format_duration_hhmmss
from speakers.services import run_diarization, run_vad
from speakers.models import SpeakerSegment, SilenceSegment, SpeakerLabel
from transcriptions.assembly import assemble_chunk_review_display

logger: Final[logging.Logger] = logging.getLogger("pipelines")

def queue_jobs(*, recording: Recording, backend: BackendName | None = None, model: ModelName | None = None, include_failed: bool = False) -> int:
    """
    Queue chunk transcription jobs for a recording.
    :arg recording: Recording whose chunks should be queued
    :arg backend: Optional backend to use for transcription. Defaults resolved by ModelManager.
    :arg model: Optional model to use for transcription. Defaults resolved by ModelManager.
    :arg include_failed: Whether failed chunks should also be re-queued
    :return: The number of chunks queued
    :raises ValueError: If the recording is not in a queueable state or has no eligible chunks.
    """
    if recording.status != RecordingStatus.CHUNKED:
        raise ValueError("recording must be in chunked status before queueing jobs")
    manager: ModelManager = ModelManager()
    selection: ResolvedModelSelection = manager.resolve_selection(backend=backend, model=model)
    allowed_statuses: list[ChunkStatus] = [ChunkStatus.PENDING]
    if include_failed:
        allowed_statuses.append(ChunkStatus.FAILED)
    chunks: list[Chunk] = list(Chunk.objects.filter(recording=recording, status__in=allowed_statuses).order_by("chunk_index"))
    if not chunks:
        raise ValueError("recording has no eligible chunks to queue")

    logger.info(
        "queue_jobs_started",
        extra={
            "recording_id": str(recording.id),
            "chunk_count": len(chunks),
            "backend": selection.backend.value,
            "model": selection.model.value,
            "include_failed": include_failed,
        }
    )

    with transaction.atomic():
        for chunk in chunks:
            chunk.status = ChunkStatus.QUEUED
            chunk.save(update_fields=["status", "updated_at"])
            enqueue_transcription_job(
                job=TranscriptionJob(chunk_id=chunk.id, backend=selection.backend, model=selection.model),
            )
        recording.status = RecordingStatus.TRANSCRIBING
        recording.save(update_fields=["status", "updated_at"])

    logger.info(
        "queue_jobs_completed",
        extra={
            "recording_id": str(recording.id),
            "enqueued_count": len(chunks),
            "status": recording.status,
        }
    )
    return len(chunks)


@dataclass(frozen=True)
class UploadAndNormalizeResult:
    """
    Result of the upload and normalize orchestration flow.
    :arg recording: The persisted recording row
    :arg normalization_succeeded: Whether normalization completed successfully
    :arg warning: Optional warning message when upload succeeded but normalization failed.
    """
    recording: Recording
    normalization_succeeded: bool
    warning: str | None = None

@dataclass(frozen=True)
class UploadNormalizedAndDiarizedResult:
    """
    Result of the upload + normalize + diarize orchestration flow.
    :arg recording: The persisted recording row
    :arg normalization_succeeded: Whether normalization completed successfully
    :arg diarization_succeeded: Whether diarization completed successfully
    :arg warning: Optional warning message when upload succeeded but normalization
    """
    recording: Recording
    normalization_succeeded: bool
    diarization_succeeded: bool
    warning: str | None = None


@dataclass(frozen=True)
class UploadNormalizedDiarizedVadResult:
    """
    Result of the upload + normalize + diarize + VAD orchestration flow.
    :arg recording: The persisted recording row
    :arg normalization_succeeded: Whether normalization completed successfully
    :arg diarization_succeeded: Whether diarization completed successfully
    :arg vad_succeeded: Whether VAD completed successfully
    :arg warning: Optional warning message when upload succeeded but normalization
    """
    recording: Recording
    normalization_succeeded: bool
    diarization_succeeded: bool
    vad_succeeded: bool
    warning: str | None = None


@dataclass(frozen=True)
class UploadNormalizeDiarizeVadAndChunkResult:
    """
    Result of the upload + normalize + diarize + VAD +chunk orchestration flow.
    :arg recording: The persisted recording row
    :arg normalization_succeeded: Whether normalization completed successfully
    :arg diarization_succeeded: Whether diarization completed successfully
    :arg vad_succeeded: Whether VAD completed successfully
    :arg warning: Optional warning message when upload succeeded but normalization
    """
    recording: Recording
    normalization_succeeded: bool
    diarization_succeeded: bool
    vad_succeeded: bool
    chunk_creation_succeeded: bool
    warning: str | None = None


@dataclass(frozen=True)
class FullPipelineResult:
    """
    Result of the upload + normalize + diarize + VAD +chunk orchestration flow.
    :arg recording: The persisted recording row
    :arg normalization_succeeded: Whether normalization completed successfully
    :arg diarization_succeeded: Whether diarization completed successfully
    :arg vad_succeeded: Whether VAD completed successfully
    :arg warning: Optional warning message when upload succeeded but normalization
    """
    recording: Recording
    normalization_succeeded: bool
    diarization_succeeded: bool
    vad_succeeded: bool
    chunk_creation_succeeded: bool
    queued_count: int
    status: str
    warning: str | None = None


@dataclass(frozen=True)
class StartWorkspacePipelineResult:
    """
    Result of starting the workspace pipeline orchestration flow.
    :arg recording: The persisted recording row created during upload ingestion.
    """
    recording: Recording


def _derive_workspace_phase(
        *,
        recording: Recording,
        speaker_count: int,
        chunk_count: int,
) -> str:
    """
    Derive the current workspace phase from persisted database state.
    :param recording: Recording instance.
    :param speaker_count: Number of persisted speaker identities.
    :param chunk_count: Number of persisted chunks.
    :return: Workspace phase identifier.
    """
    if recording.status == RecordingStatus.FAILED:
        return "failed"
    if chunk_count > 0 or recording.status in {RecordingStatus.CHUNKED, RecordingStatus.TRANSCRIBING, RecordingStatus.COMPLETED}:
        return "chunk_review"
    if speaker_count > 0 or recording.status == RecordingStatus.DIARIZED:
        return "speakers_ready"
    if recording.normalized_file_path:
        return "diarizing"
    return "preparing"


def get_workspace_state(*, recording_id: UUID) -> dict[str, object]:
    """
    Return the assembled workspace snapshot for one recording.
    :param recording_id: UUID of the recording.
    :return: Workspace state payload for initial render and reconnect flows.
    """
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist as exc:
        raise ValueError(f"Recording {recording_id} was not found") from exc

    speaker_labels_by_id: dict[str, str] = {
        label.speaker_id: label.display_name
        for label in SpeakerLabel.objects.filter(recording=recording).order_by("speaker_id")
    }
    speaker_ids: list[str] = list(
        SpeakerSegment.objects.filter(recording=recording).order_by("start_time", "end_time").values_list("speaker_id", flat=True).distinct()
    )
    speaker_items: list[dict[str, str | None]] = [
        {
            "speaker_id": speaker_id,
            "display_name": speaker_labels_by_id.get(speaker_id),
        }
        for speaker_id in speaker_ids
    ]
    silence_segment_count = SilenceSegment.objects.filter(recording=recording).count()
    chunks: list[Chunk] = list(
        Chunk.objects.filter(recording=recording).select_related("transcript").order_by("chunk_index")
    )
    next_chunks_by_id: dict[str, Chunk | None] = {
        str(chunk.id): chunks[index + 1] if index + 1 < len(chunks) else None
        for index, chunk in enumerate(chunks)
    }
    chunk_items: list[dict[str, object]] = []
    for chunk in chunks:
        has_transcript = hasattr(chunk, "transcript")
        chunk_items.append(
            {
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "status": chunk.status,
                "has_transcript": has_transcript,
                "accepted_text": chunk.transcript.accepted_text if has_transcript else "",
                "review_display": (
                    assemble_chunk_review_display(
                        chunk=chunk,
                        next_chunk=next_chunks_by_id[str(chunk.id)],
                    )
                    if has_transcript and chunk.status in {ChunkStatus.COMPLETED, ChunkStatus.NEEDS_REVIEW}
                    else None
                ),
            }
        )
    chunk_count = len(chunks)
    workspace_phase = _derive_workspace_phase(
        recording=recording,
        speaker_count=len(speaker_ids),
        chunk_count=chunk_count,
    )

    return {
        "recording_id": str(recording.id),
        "recording_status": recording.status,
        "workspace_phase": workspace_phase,
        "recording": {
            "original_file_name": recording.original_file_name,
            "duration_milliseconds": recording.duration_milliseconds,
            "formatted_duration": format_duration_hhmmss(duration_milliseconds=recording.duration_milliseconds),
            "has_normalized_audio": bool(recording.normalized_file_path),
            "normalized_audio_url": (
                reverse("recordings:normalized_audio", args=[recording.id])
                if recording.normalized_file_path
                else None
            ),
        },
        "speakers": {
            "count": len(speaker_ids),
            "items": speaker_items,
        },
        "vad": {
            "silence_segment_count": silence_segment_count,
            "completed": silence_segment_count > 0,
        },
        "chunks": {
            "total": chunk_count,
            "pending": sum(1 for chunk in chunks if chunk.status == ChunkStatus.PENDING),
            "queued": sum(1 for chunk in chunks if chunk.status == ChunkStatus.QUEUED),
            "processing": sum(1 for chunk in chunks if chunk.status == ChunkStatus.PROCESSING),
            "completed": sum(1 for chunk in chunks if chunk.status == ChunkStatus.COMPLETED),
            "failed": sum(1 for chunk in chunks if chunk.status == ChunkStatus.FAILED),
            "needs_review": sum(1 for chunk in chunks if chunk.status == ChunkStatus.NEEDS_REVIEW),
            "items": chunk_items,
        },
    }


def run_workspace_pipeline(*, recording_id: UUID) -> None:
    """
    Run the workspace preprocessing pipeline for one recording.

    This orchestration flow executes the current preprocessing stages:
    1. normalizes the uploaded recording
    2. runs diarization
    3. runs VAD
    4. creates chunks
    5. queues transcription jobs
    :param recording_id: UUID of the recording whose pipeline should run.
    :return: None
    """
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist as exc:
        raise ValueError(f"Recording {recording_id} was not found") from exc

    logger.info(
        "workspace_pipeline_processing_started",
        extra={
            "recording_id": str(recording_id),
            "status": recording.status,
            "steps": [
                "normalize_audio",
                "run_diarization",
                "run_vad",
                "create_chunks",
                "queue_jobs",
            ],
        }
    )
    publish_workspace_event(
        recording_id=recording.id,
        event_type="pipeline_started",
        payload={
            "workspace_phase": "preparing",
            "recording_status": recording.status,
        },
    )

    try:
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_started",
            payload={
                "step": "normalization",
                "workspace_phase": "preparing",
            },
        )
        recording = normalize_audio(recording=recording)
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_completed",
            payload={
                "step": "normalization",
                "workspace_phase": "preparing",
                "recording_status": recording.status,
            },
        )
    except Exception as exc:
        warning = "Upload succeeded but normalization failed."
        logger.warning(
            "workspace_pipeline_normalization_failed_after_upload",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="pipeline_failed",
            payload={
                "step": "normalization",
                "workspace_phase": "preparing",
                "recording_status": recording.status,
                "message": warning,
            },
        )
        raise

    try:
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_started",
            payload={
                "step": "diarization",
                "workspace_phase": "diarizing",
            },
        )
        run_diarization(recording=recording)
        recording.refresh_from_db()
        speaker_ids: list[str] = list(
            SpeakerSegment.objects.filter(recording=recording).order_by("start_time", "end_time").values_list("speaker_id", flat=True).distinct()
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="speakers_detected",
            payload={
                "workspace_phase": "speakers_ready",
                "recording_status": recording.status,
                "speaker_count": len(speaker_ids),
                "speaker_ids": speaker_ids,
            },
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_completed",
            payload={
                "step": "diarization",
                "workspace_phase": "speakers_ready",
                "recording_status": recording.status,
            },
        )
    except Exception as exc:
        recording.refresh_from_db()
        warning = "Upload and normalization succeeded but diarization failed."
        logger.warning(
            "workspace_pipeline_diarization_failed_after_normalization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="pipeline_failed",
            payload={
                "step": "diarization",
                "workspace_phase": "diarizing",
                "recording_status": recording.status,
                "message": warning,
            },
        )
        raise

    try:
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_started",
            payload={
                "step": "vad",
                "workspace_phase": "speakers_ready",
            },
        )
        run_vad(recording=recording)
        silence_segment_count = SilenceSegment.objects.filter(recording=recording).count()
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_completed",
            payload={
                "step": "vad",
                "workspace_phase": "speakers_ready",
                "recording_status": recording.status,
                "silence_segment_count": silence_segment_count,
            },
        )
    except Exception as exc:
        recording.refresh_from_db()
        warning = "Upload, normalization, and diarization succeeded but VAD failed."
        logger.warning(
            "workspace_pipeline_vad_failed_after_diarization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="pipeline_failed",
            payload={
                "step": "vad",
                "workspace_phase": "speakers_ready",
                "recording_status": recording.status,
                "message": warning,
            },
        )
        raise

    try:
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_started",
            payload={
                "step": "chunking",
                "workspace_phase": "speakers_ready",
            },
        )
        create_chunks(recording=recording)
        recording.refresh_from_db()
        total_chunks = Chunk.objects.filter(recording=recording).count()
        publish_workspace_event(
            recording_id=recording.id,
            event_type="chunks_created",
            payload={
                "workspace_phase": "chunk_review",
                "recording_status": recording.status,
                "total_chunks": total_chunks,
            },
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_completed",
            payload={
                "step": "chunking",
                "workspace_phase": "chunk_review",
                "recording_status": recording.status,
                "total_chunks": total_chunks,
            },
        )
    except Exception as exc:
        recording.refresh_from_db()
        warning = "Upload, normalization, diarization, and VAD succeeded but chunk creation failed."
        logger.warning(
            "workspace_pipeline_chunk_creation_failed_after_vad",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="pipeline_failed",
            payload={
                "step": "chunking",
                "workspace_phase": "speakers_ready",
                "recording_status": recording.status,
                "message": warning,
            },
        )
        raise

    recording.refresh_from_db()

    try:
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_started",
            payload={
                "step": "queue_jobs",
                "workspace_phase": "chunk_review",
            },
        )
        queue_jobs(recording=recording)
        recording.refresh_from_db()
        chunk_progress_payload = get_recording_process(recording_id=recording.id)
        publish_workspace_event(
            recording_id=recording.id,
            event_type="chunk_progress",
            payload=chunk_progress_payload,
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="step_completed",
            payload={
                "step": "queue_jobs",
                "workspace_phase": "chunk_review",
                "recording_status": recording.status,
            },
        )
    except Exception as exc:
        recording.refresh_from_db()
        warning = "Upload, normalization, diarization, VAD, and chunk creation succeeded but queueing failed."
        logger.warning(
            "workspace_pipeline_queueing_failed_after_chunk_creation",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        publish_workspace_event(
            recording_id=recording.id,
            event_type="pipeline_failed",
            payload={
                "step": "queue_jobs",
                "workspace_phase": "chunk_review",
                "recording_status": recording.status,
                "message": warning,
            },
        )
        raise

    logger.info(
        "workspace_pipeline_processing_completed",
        extra={
            "recording_id": str(recording_id),
            "status": recording.status,
        }
    )
    publish_workspace_event(
        recording_id=recording.id,
        event_type="pipeline_completed",
        payload={
            "workspace_phase": "chunk_review",
            "recording_status": recording.status,
        },
    )


def start_workspace_pipeline(*, uploaded_file: UploadedFile) -> StartWorkspacePipelineResult:
    """
    Ingest an uploaded recording and trigger the workspace preprocessing pipeline asynchronously.

    The synchronous portion of this workflow is intentionally limited to:
    1. Persist the original upload
    2. Create the recording row in the database
    3. Enqueue the workspace background pipeline
    :param uploaded_file: Django uploaded file object.
    :return: StartWorkspacePipelineResult containing the created recording row.
    :raises ValueError: If the uploaded file is invalid.
    :raises Exception: Any ingestion error prior to successful recording creation is propagated.
    """
    recording = ingest_uploaded_recording(uploaded_file=uploaded_file)

    logger.info(
        "workspace_pipeline_start_completed",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "original_file_name": recording.original_file_name,
        }
    )

    enqueue_workspace_pipeline_job(
        job=WorkspacePipelineJob(recording_id=recording.id),
    )

    return StartWorkspacePipelineResult(recording=recording)


def upload_and_normalize_recording(*, uploaded_file: UploadedFile) -> UploadAndNormalizeResult:
    """
    Ingest an uploaded recording and then attempt audio normalization. This function orchestrates the firs two stages
    of the processing pipeline:
    1. Upload recording -> store file -> create DN row
    2. Normalize recording
    - If ingestion fails before the `Recording` row exists, the exception is
      propagated and any partial file artifacts should already have been cleaned.
    - If normalization fails after the `Recording` row exists, the original file
      and DB row are preserved and the function returns a partial-success result.
    :param uploaded_file: Django uploaded file object.
    :return: UploadAndNormalizeResult: Result object containing the persisted recording and normalization state.
    :raises ValueError: If the uploaded file is invalid.
    :raises Exception: Any ingestion error prior to successful recording creation is propagated.
    """
    recording =  ingest_uploaded_recording(uploaded_file=uploaded_file)
    try:
        normalized_recording = normalize_audio(recording=recording)
    except Exception as exc:
        warning = "Upload succeeded but normalization failed."
        logger.warning(
            "recording_normalization_failed_after_upload",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadAndNormalizeResult(
            recording=recording,
            normalization_succeeded=False,
            warning=warning,
        )
    return UploadAndNormalizeResult(
        recording=normalized_recording,
        normalization_succeeded=True,
        warning=None,
    )

def upload_normalize_and_diarize_recording(*, uploaded_file: UploadedFile) -> UploadNormalizedAndDiarizedResult:
    """
    Upload a recording, normalize its audio, and run full-recording diarization.

    1. Upload recording -> store file -> create DN row
    2. Normalize recording
    3. Run diarization and persist SpeakerSegment rows
    :param uploaded_file: Django uploaded file object.
    :return: UploadNormalizedAndDiarizedResult: Structured result containing the recording and stage outcomes.
    """
    recording = ingest_uploaded_recording(uploaded_file=uploaded_file)
    try:
        recording = normalize_audio(recording=recording)
    except Exception as exc:
        warning = "Upload succeeded but normalization failed."
        logger.warning(
            "pipeline_normalization_failed_after_upload",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            },
        )
        return UploadNormalizedAndDiarizedResult(
            recording=recording,
            normalization_succeeded=False,
            diarization_succeeded=False,
            warning=warning,
        )
    try:
        run_diarization(recording=recording)
    except Exception as exc:
        recording.refresh_from_db()
        warning = "Upload and normalization succeeded but diarization failed."
        logger.warning(
            "pipeline_diarization_failed_after_normalization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            },
        )
        return UploadNormalizedAndDiarizedResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=False,
            warning=warning,
        )
    recording.refresh_from_db()
    logger.info(
        "pipeline_upload_normalize_and_diarize_completed",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "normalization_succeeded": True,
            "diarization_succeeded": True,
        },
    )
    return UploadNormalizedAndDiarizedResult(
        recording=recording,
        normalization_succeeded=True,
        diarization_succeeded=True,
        warning=None,
    )

def upload_normalize_diarize_and_vad_recording(*, uploaded_file: UploadedFile) -> UploadNormalizedDiarizedVadResult:
    """
    Upload a recording and run the following pipeline stages:

    1. Upload recording -> store file -> create DN row
    2. Normalize recording
    3. Run diarization and persist SpeakerSegment rows
    4. run VAD
    :param uploaded_file: Django uploaded file object.
    :return: UploadNormalizedAndDiarizedResult: Structured result containing the recording and stage outcomes.
    """
    recording = ingest_uploaded_recording(uploaded_file=uploaded_file)

    try:
        recording = normalize_audio(recording=recording)
    except Exception as exc:
        warning = "Upload succeeded but normalization failed."
        logger.warning(
            "pipeline_normalization_failed_after_upload",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return  UploadNormalizedDiarizedVadResult(
            recording=recording,
            normalization_succeeded=False,
            diarization_succeeded=False,
            vad_succeeded=False,
            warning=warning,
        )
    try:
        run_diarization(recording=recording)
    except Exception as exc:
        warning = "Upload and normalization succeeded but diarization failed."
        logger.warning(
            "pipeline_diarization_failed_after_normalization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadNormalizedDiarizedVadResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=False,
            vad_succeeded=False,
            warning=warning,
        )

    try:
        run_vad(recording=recording)
    except Exception as exc:
        warning = "Upload, normalization, and diarization succeeded but VAD failed."
        logger.warning(
            "pipeline_vad_failed_after_diarization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadNormalizedDiarizedVadResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=True,
            vad_succeeded=False,
            warning=warning,
        )

    recording.refresh_from_db()

    logger.info(
        "pipeline_upload_normalize_diarize_and_vad_completed",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "normalization_succeeded": True,
            "diarization_succeeded": True,
            "vad_succeeded": True,
        }
    )
    return UploadNormalizedDiarizedVadResult(
        recording=recording,
        normalization_succeeded=True,
        diarization_succeeded=True,
        vad_succeeded=True,
        warning=None,
    )

def upload_normalize_diarize_vad_and_chunk_recording(
        *,
        uploaded_file: UploadedFile,
        chunk_duration_milliseconds: int = 30_000,
        overlap_milliseconds: int = 5_000,
) -> UploadNormalizeDiarizeVadAndChunkResult:
    """
    Upload a recording and run the following pipeline stages:

    1. Upload recording -> store file -> create DN row
    2. Normalize recording
    3. Run diarization and persist SpeakerSegment rows
    4. run VAD
    5. Create chunks
    :param uploaded_file: Django uploaded file object.
    :param chunk_duration_milliseconds: Duration of chunk in milliseconds.
    :param overlap_milliseconds: Duration of overlap in milliseconds.
    :return: UploadNormalizeDiarizeVadAndChunkResult: Structured result containing the recording and stage outcomes.
    """
    recording = ingest_uploaded_recording(uploaded_file=uploaded_file)
    try:
        recording = normalize_audio(recording=recording)
    except Exception as exc:
        warning = "Upload succeeded but normalization failed."
        logger.warning(
            "pipeline_normalization_failed_after_upload",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadNormalizeDiarizeVadAndChunkResult(
            recording=recording,
            normalization_succeeded=False,
            diarization_succeeded=False,
            vad_succeeded=False,
            chunk_creation_succeeded=False,
            warning=warning,
        )
    try:
        run_diarization(recording=recording)
    except Exception as exc:
        warning = "Upload and normalization succeeded but diarization failed."
        logger.warning(
            "pipeline_diarization_failed_after_normalization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadNormalizeDiarizeVadAndChunkResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=False,
            vad_succeeded=False,
            chunk_creation_succeeded=False,
            warning=warning,
        )

    try:
        run_vad(recording=recording)
    except Exception as exc:
        warning = "Upload, normalization, and diarization succeeded but VAD failed."
        logger.warning(
            "pipeline_vad_failed_after_diarization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadNormalizeDiarizeVadAndChunkResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=True,
            vad_succeeded=False,
            chunk_creation_succeeded=False,
            warning=warning,
        )
    try:
        create_chunks(
            recording=recording,
            chunk_duration_milliseconds=chunk_duration_milliseconds,
            overlap_milliseconds=overlap_milliseconds,
        )
    except Exception as exc:
        warning = "Upload, normalization, diarization, and VAD succeeded but chunk creation failed."
        logger.warning(
            "pipeline_chunk_creation_failed_after_vad",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return UploadNormalizeDiarizeVadAndChunkResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=True,
            vad_succeeded=True,
            chunk_creation_succeeded=False,
            warning=warning,
        )
    recording.refresh_from_db()
    logger.info(
        "pipeline_upload_normalize_diarize_vad_and_chunk_completed",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "normalization_succeeded": True,
            "diarization_succeeded": True,
            "vad_succeeded": True,
            "chunk_creation_succeeded": True,
        }
    )
    return UploadNormalizeDiarizeVadAndChunkResult(
        recording=recording,
        normalization_succeeded=True,
        diarization_succeeded=True,
        vad_succeeded=True,
        chunk_creation_succeeded=True,
        warning=None,
    )

def start_full_transcription(*, uploaded_file: UploadedFile) ->FullPipelineResult:
    """
    Queue all eligible chunks for a recording and transition it into transcribing.
    :param uploaded_file: Django uploaded file object.
    :return: Summary payload for the caller.
    """
    recording = ingest_uploaded_recording(uploaded_file=uploaded_file)
    try:
        recording = normalize_audio(recording=recording)
    except Exception as exc:
        warning = "Upload succeeded but normalization failed."
        logger.warning(
            "pipeline_normalization_failed_after_upload",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return FullPipelineResult(
            recording=recording,
            normalization_succeeded=False,
            diarization_succeeded=False,
            vad_succeeded=False,
            chunk_creation_succeeded=False,
            queued_count=0,
            status=recording.status,
            warning=warning,
        )
    try:
        run_diarization(recording=recording)
    except Exception as exc:
        warning = "Upload and normalization succeeded but diarization failed."
        logger.warning(
            "pipeline_diarization_failed_after_normalization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return FullPipelineResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=False,
            vad_succeeded=False,
            chunk_creation_succeeded=False,
            queued_count=0,
            status=recording.status,
            warning=warning,
        )

    try:
        run_vad(recording=recording)
    except Exception as exc:
        warning = "Upload, normalization, and diarization succeeded but VAD failed."
        logger.warning(
            "pipeline_vad_failed_after_diarization",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return FullPipelineResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=True,
            vad_succeeded=False,
            chunk_creation_succeeded=False,
            queued_count=0,
            status=recording.status,
            warning=warning,
        )
    try:
        create_chunks(recording=recording)
    except Exception as exc:
        warning = "Upload, normalization, diarization, and VAD succeeded but chunk creation failed."
        logger.warning(
            "pipeline_chunk_creation_failed_after_vad",
            extra={
                "recording_id": str(recording.id),
                "status": recording.status,
                "warning": warning,
                "error": str(exc),
            }
        )
        return FullPipelineResult(
            recording=recording,
            normalization_succeeded=True,
            diarization_succeeded=True,
            vad_succeeded=True,
            chunk_creation_succeeded=False,
            queued_count=0,
            status=recording.status,
            warning=warning,
        )

    recording.refresh_from_db()
    logger.info(
        "pipeline_upload_normalize_diarize_vad_and_chunk_completed",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "normalization_succeeded": True,
            "diarization_succeeded": True,
            "vad_succeeded": True,
            "chunk_creation_succeeded": True,
        }
    )

    queued_count = queue_jobs(recording=recording)

    return FullPipelineResult(
        recording=recording,
        normalization_succeeded=True,
        diarization_succeeded=True,
        vad_succeeded=True,
        chunk_creation_succeeded=True,
        queued_count=queued_count,
        status=recording.status,
    )

def get_recording_process(*, recording_id: UUID) -> dict[str, int | str]:
    """
    Return chunk-level progress summary for a recording.
    :param recording_id: UUID of the recording
    :return: Summary payload for the caller.
    """
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist as exc:
        raise ValueError(f"Recording {recording_id} was not found") from exc
    chunks = Chunk.objects.filter(recording=recording)
    total_chunks = chunks.count()
    pending_chunks = chunks.filter(status=ChunkStatus.PENDING).count()
    queued_chunks = chunks.filter(status=ChunkStatus.QUEUED).count()
    processing_chunks = chunks.filter(status=ChunkStatus.PROCESSING).count()
    completed_chunks = chunks.filter(status=ChunkStatus.COMPLETED).count()
    failed_chunks = chunks.filter(status=ChunkStatus.FAILED).count()

    return {
        "recording_id": str(recording.id),
        "recording_status": recording.status,
        "total_chunks": total_chunks,
        "pending_chunks": pending_chunks,
        "queued_chunks": queued_chunks,
        "processing_chunks": processing_chunks,
        "completed_chunks": completed_chunks,
        "failed_chunks": failed_chunks,
    }
