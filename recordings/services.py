import logging
from pathlib import Path
from typing import Final

from django.db import transaction

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