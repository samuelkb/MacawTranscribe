import logging
from pathlib import Path
from typing import Final

from django.db import transaction

from recordings.models import Recording, RecordingStatus
from speakers.audio import get_diarization_model_name, run_pyannote_diarization, DiarizationSegment
from speakers.models import SpeakerSegment

logger: Final[logging.Logger] = logging.getLogger(__name__)

def run_diarization(*, recording: Recording) -> list[SpeakerSegment]:
    """
    Run full-recording  speaker diarization for a normalized recording and persist the resulting speaker segments.

    This function replaces any previously stored speaker segments for the recording, updates recording status to
    ``diarized`` after successful persistence.
    :param recording: Recording instance to diarize
    :return: list of persisted speaker segments
    :raises: ValueError: if recording is missing a normalized file path.
    :raises: FileNotFoundError: if the normalized file does not exist.
    :raises: DiarizationError: if the diarization backend fails.
    """
    if not recording.normalized_file_path or not recording.normalized_file_path.strip():
        raise ValueError('recording.normalized_file_path must not be empty.')
    logger.info(
        "recording_diarization_started",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "normalized_file_path": recording.normalized_file_path,
        },
    )
    model_used: str = get_diarization_model_name()
    diarization_segments: list[DiarizationSegment] = run_pyannote_diarization(
        audio_path=Path(recording.normalized_file_path),
    )

    with transaction.atomic():
        SpeakerSegment.objects.filter(recording=recording).delete()

        created_segments: list[SpeakerSegment] = []
        for segment in diarization_segments:
            created_segments.append(
                SpeakerSegment.objects.create(
                    recording=recording,
                    speaker_id=segment.speaker_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    model_used=model_used,
                )
            )
        recording.status = RecordingStatus.DIARIZED
        recording.save(update_fields=["status", "updated_at"])
    logger.info(
        "recording_diarization_completed",
        extra={
            "recording_id": str(recording.id),
            "status": recording.status,
            "segment_count": len(created_segments),
            "model_used": model_used,
        },
    )
    return created_segments