import logging
import subprocess
from dataclasses import dataclass
from uuid import UUID, uuid4
from pathlib import Path
from typing import Final

from django.core.files.uploadedfile import UploadedFile
from django.db import transaction

from django.conf import settings
from recordings.audio import _run_ffmpeg_normalization, probe_audio_duration_milliseconds
from recordings.files import build_original_file_path, save_uploaded_file_atomic
from recordings.models import Recording, RecordingStatus, Chunk, ChunkStatus

logger: Final[logging.Logger] = logging.getLogger(__name__)


def format_duration_hhmmss(*, duration_milliseconds: int | None) -> str:
    """
    Format a millisecond duration as HH:MM:SS for display.
    """
    if duration_milliseconds in (None, ""):
        total_seconds = 0
    else:
        total_seconds = max(int(duration_milliseconds) // 1000, 0)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_recording(
        *,
        recording_id: UUID | None = None,
        original_file_name: str,
        original_file_path: str,
        duration_milliseconds: int
) -> Recording:
    """
    Create a new recording entry in the database.

    This function is the first step of the ingestion pipeline. It persists the uploaded recording metadata and marks
    the recording as `uploaded`.
    :param recording_id: Optional pre-generated recording ID. When omitted, a new UUID is created.
    :param original_file_name: The original filename provided by the user at upload time.
    :param original_file_path: The filesystem path where the original uploaded file was stored.
    :param duration_milliseconds: Total recording duration in milliseconds.
    :return: The newly created Recording instance,
    :raises ValueError: If the filename is empty, the path is empty, or duration is invalid.
    """
    filename = original_file_name.strip()
    filepath = original_file_path.strip()

    if not filename:
        raise ValueError("original_file_name must not be empty")
    if not filepath:
        raise ValueError("original_file_path must not be empty")
    if duration_milliseconds <= 0:
        raise ValueError("duration_milliseconds must be greater than 0")

    path_object = Path(filename)
    if path_object.name == "":
        raise ValueError("original_file_path must point to a file")

    with transaction.atomic():
        recording = Recording.objects.create(
            id= recording_id or uuid4(),
            original_file_name=filename,
            original_file_path=filepath,
            duration_milliseconds=duration_milliseconds,
            status=RecordingStatus.UPLOADED,
        )

    logger.info(
        "Recording created successfully.",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "original_file_name": recording.original_file_name,
            "duration_milliseconds": recording.duration_milliseconds,
        },
    )

    return recording


def normalize_audio(*, recording: Recording) -> Recording:
    """
    Normalize the original audio for a recording and persist the normalized path.

    This converts the uploaded source media into the normalized working format:
    - WAV container
    - 16kHz sample rate
    - mono channel
    - PCM 16-bit audio
    :param recording: The Recording instance to be normalized.
    :return: The updated Recording instance.
    :raises ValueError: If the recording does not have a valid original file path.
    :raises FileNotFoundError: If the original file path does not exist on disk.
    :raises AudioNormalizationError: If ffmpeg fails to normalize the audio file.
    """
    if not recording.original_file_path or not recording.original_file_path.strip():
        raise ValueError("recording.original_file_path must not be empty")

    input_path = Path(recording.original_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"original audio file was not found: {input_path}")

    recording_dir = input_path.parent
    output_path = recording_dir / "normalized.wav"

    logger.info(
        "audio normalization started.",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "original_file_name": recording.original_file_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg_normalization(input_path=input_path, output_path=output_path)

    with transaction.atomic():
        recording.normalized_file_path = str(output_path)
        recording.status = RecordingStatus.NORMALIZED
        recording.save(update_fields=["normalized_file_path", "status", "updated_at"])

    logger.info(
        "audio normalization completed.",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "normalized_file_path": recording.normalized_file_path,
        }
    )

    return recording

def get_recordings_base_dir() -> Path:
    """
    Resolve the base directory where recording files are stored.
    :return: Path: Base path for recording storage
    """
    configured = getattr(settings, "RECORDINGS_BASE_DIR", "data/recordings")
    return Path(configured)

def _cleanup_failed_ingestion(*, saved_path: Path | None) -> None:
    """
    Clean up for a failed recording ingestion. If ingestion fails before the Recording row is created, remove any
    saved file and remove its directory if it becomes empty.
    :param saved_path:
    :return: None
    """
    if saved_path is None:
        return
    try:
        if saved_path.exists():
            saved_path.unlink()
        recording_dir = saved_path.parent
        if recording_dir.exists():
            try:
                recording_dir.rmdir()
            except OSError:
                # Directory not empty
                pass
    except Exception:
        logger.exception(
            "recording_ingestion_cleanup_failed",
            extra={
                "saved_path": str(saved_path),
            }
        )

def ingest_uploaded_recording(*, uploaded_file: UploadedFile) -> Recording:
    """
    Ingest an uploaded recording into the system. This orchestrates the initial upload pipeline:
    1. Generate a recording ID
    2. Persist the original uploaded file to disk
    3. Probe the audio duration
    4. Create the Recording row in the database

    On failure before DB creation completes, the saved file is cleaned up.
    :param uploaded_file: Django uploaded file object received from a request
    :return: Recording: The created Recording instance
    :raises ValueError: If the uploaded file is invalid.
    :raises UploadedFileSaveError: If the original file cannot be persisted.
    :raises AudioProbeError: If the duration cannot be probed.
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file must not be None")
    original_file_name = (uploaded_file.name or "").strip()
    if not original_file_name:
        raise ValueError("uploaded_file.name must not be empty")

    recording_id = uuid4()
    base_dir = get_recordings_base_dir()
    original_file_path = build_original_file_path(
        recording_id=recording_id,
        original_file_name=original_file_name,
        base_dir=base_dir,
    )

    logger.info("recording_ingestion_started",
                extra={
                    "recording_id": str(recording_id),
                    "original_file_name": original_file_name,
                    "destination_path": str(base_dir),
                }
    )

    saved_path: Path | None = None
    try:
        saved_path = save_uploaded_file_atomic(
            uploaded_file=uploaded_file,
            destination_path=original_file_path,
        )
        duration_milliseconds = probe_audio_duration_milliseconds(input_path=saved_path)
        recording = create_recording(
            recording_id=recording_id,
            original_file_name=original_file_name,
            original_file_path=str(saved_path),
            duration_milliseconds=duration_milliseconds
        )
    except Exception:
        _cleanup_failed_ingestion(saved_path=saved_path)
        raise
    logger.info("recording_ingestion_completed",
                extra={
                    "recording_id": str(recording_id),
                    "original_file_name": recording.original_file_name,
                    "duration_milliseconds": recording.duration_milliseconds,
                    "status": recording.status,
                }
    )

    return recording

def create_chunks(
        *,
        recording: Recording,
        chunk_duration_milliseconds: int = 30_000,
        overlap_milliseconds: int = 5_000
) -> list[Chunk]:
    """
    Create fixed-duration chunk metadata for a recording.

    Chunks are processing units used later for queueing and transcription. This function creates metadata only,
    it does not extract chunk audio files. Existing chunks for the recording are replaced.
    :param recording: Recording instance to chunk
    :param chunk_duration_milliseconds: Target Chunk duration in milliseconds
    :param overlap_milliseconds: Overlap between consecutive chunks in milliseconds
    :return: list[Chunk]: Persisted chunk rows ordered by chunk_index
    :raises ValueError: If recording is missing normalized audio, has invalid duration, or if chunk sizing parameters
    are invalid.
    """
    if not recording.normalized_file_path or not recording.normalized_file_path.strip():
        raise ValueError("recording.normalized_file_path must not be empty")
    if recording.duration_milliseconds <= 0:
        raise ValueError("recording.duration_milliseconds must be greater than 0")
    if chunk_duration_milliseconds <= 0:
        raise ValueError("chunk_duration_milliseconds must be greater than 0")
    if overlap_milliseconds < 0:
        raise ValueError("overlap_milliseconds must not be negative")
    if overlap_milliseconds >= chunk_duration_milliseconds:
        raise ValueError("overlap_milliseconds must be smaller than chunk_duration_milliseconds")

    step_milliseconds = chunk_duration_milliseconds - overlap_milliseconds
    logger.info(
        "chunk_creation_started",
        extra={
            "recording_id": str(recording.id),
            "duration_milliseconds": recording.duration_milliseconds,
            "chunk_duration_milliseconds": chunk_duration_milliseconds,
            "overlap_milliseconds": overlap_milliseconds,
            "step_milliseconds": step_milliseconds,
        }
    )

    chunks_to_create: list[Chunk] = []
    chunk_index: int = 0
    start_time: int = 0

    while start_time < recording.duration_milliseconds:
        end_time: int = min(start_time + chunk_duration_milliseconds, recording.duration_milliseconds)

        chunks_to_create.append(
            Chunk(
                recording=recording,
                chunk_index=chunk_index,
                start_time=start_time,
                end_time=end_time,
                status=ChunkStatus.PENDING,
            )
        )

        if end_time >= recording.duration_milliseconds:
            break

        chunk_index += 1
        start_time += step_milliseconds

    with transaction.atomic():
        Chunk.objects.filter(recording=recording).delete()
        created_chunks: list[Chunk] = Chunk.objects.bulk_create(chunks_to_create)
        recording.status = RecordingStatus.CHUNKED
        recording.save(update_fields=["status", "updated_at"])

    logger.info(
        "chunk_creation_completed",
        extra={
            "recording_id": str(recording.id),
            "chunk_count": len(created_chunks),
            "status": recording.status,
        }
    )
    return created_chunks
