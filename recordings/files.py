import logging
import os
import tempfile
from pathlib import Path
from typing import Final
from uuid import UUID

from django.core.files.uploadedfile import UploadedFile

logger: Final[logging.Logger] = logging.getLogger(__name__)

class UploadedFileSaveError(RuntimeError):
    """Raised when saving an uploaded file to disk fails."""


def build_recording_directory(*, recording_id: UUID, base_dir: Path) -> Path:
    """
    Build the on-disk directory path for a recording
    :param recording_id: The unique recording identifier
    :param base_dir: Base recordings directory
    :return: Path: The target directory for the recording
    """
    return base_dir / str(recording_id)

def build_original_file_path(*, recording_id: UUID, original_file_name:str, base_dir: Path) -> Path:
    """
    Build the destination path for the original uploaded recording. The stored filename is normalized to
    ``original.<ext>`` so the filesystem layout remains deterministic and independent of user-provided names.
    :param recording_id: The unique recording identifier
    :param original_file_name: User-provided uploaded file name
    :param base_dir: Base recordings directory
    :return: Path: Final path where the uploaded file should be stored
    """
    suffix = Path(original_file_name).suffix.lower() or ".bin"
    recording_dir = build_recording_directory(recording_id=recording_id, base_dir=base_dir)
    return recording_dir / f"original{suffix}"

def save_uploaded_file_atomic(*, uploaded_file: UploadedFile, destination_path: Path) -> Path:
    """
    Persist an uploaded file atomically. The file is first written to a temporary file in the destination directory
    and then atomically renamed to the final path. This reduces the chance of leaving partially written final files
    after interruptions.
    :param uploaded_file: Django uploaded file object
    :param destination_path: Final target path for the stored file
    :return: The final stored file path
    :raises ValueError: If the uploaded file or destination path is invalid
    :raises UploadedFileSaveError: If saving the file fails
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file cannot be None")
    if not str(destination_path).strip():
        raise ValueError("destination_path cannot be empty")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file_handle = None
    temp_file_path: Path | None = None

    logger.info("uploaded_file_save_started",
                extra={
                    "destination_path": destination_path,
                    "uploaded_file": uploaded_file.name
                }
    )

    try:
        temp_file_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_path.parent,
            prefix=".upload-",
            suffix=".tmp",
            delete=False
        )
        temp_file_path = Path(temp_file_handle.name)

        for chunk in uploaded_file.chunks():
            temp_file_handle.write(chunk)

        temp_file_handle.flush()
        os.fsync(temp_file_handle.fileno())
        temp_file_handle.close()
        temp_file_handle = None

        os.replace(temp_file_path, destination_path)
    except Exception as exc:
        if temp_file_handle is not None and not temp_file_handle.closed:
            temp_file_handle.close()
        if temp_file_path is not None and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        raise UploadedFileSaveError(f"failed to save upladed file to {destination_path}") from exc

    logger.info("uploaded_file_save_completed",
                extra={
                    "destination_path": str(destination_path),
                    "uploaded_file_name": uploaded_file.name,
                    "size_bytes": getattr(uploaded_file, "size", None),
                }
    )
    return destination_path