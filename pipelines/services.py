import logging
from typing import Final
from dataclasses import dataclass
from uuid import UUID

from django.core.files.uploadedfile import UploadedFile
from django.db import transaction

from ml.manager import ModelManager, ResolvedModelSelection
from ml.types import BackendName, ModelName
from pipelines.queue import enqueue_transcription_job
from pipelines.queue_types import TranscriptionJob
from recordings.models import Recording, RecordingStatus, ChunkStatus, Chunk
from recordings.services import ingest_uploaded_recording, normalize_audio, create_chunks
from speakers.services import run_diarization, run_vad

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