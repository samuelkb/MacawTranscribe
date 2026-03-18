import logging
import subprocess
from pathlib import Path
from typing import Final

from django.db import transaction

from recordings.audio import _run_ffmpeg_normalization
from recordings.models import Recording, RecordingStatus

logger: Final[logging.Logger] = logging.getLogger("app.recordings")

def create_recording(*, original_file_name: str, original_file_path: str, duration_milliseconds: int) -> Recording:
    """
    Create a new recording entry in the database.

    This function is the first step of the ingestion pipeline. It persists the uploaded recording metadata and marks
    the recording as `uploaded`.
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
    output_path = recording_dir / f"{recording.original_file_name}_normalized.wav"

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