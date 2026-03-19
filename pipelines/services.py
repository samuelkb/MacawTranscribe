import logging
from typing import Final
from dataclasses import dataclass

from django.core.files.uploadedfile import UploadedFile

from recordings.models import Recording
from recordings.services import ingest_uploaded_recording, normalize_audio

logger: Final[logging.Logger] = logging.getLogger("pipelines")

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