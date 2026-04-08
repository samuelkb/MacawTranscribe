import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from django.db import transaction

from recordings.models import Recording, RecordingStatus
from speakers.audio import get_diarization_model_name, run_pyannote_diarization, DiarizationSegment, VadSpeechSegment, \
    get_vad_model_name, run_vad_backend
from speakers.models import SpeakerSegment, SilenceSegment, SpeakerLabel

logger: Final[logging.Logger] = logging.getLogger(__name__)

@dataclass(frozen=True)
class SilenceInterval:
    """
    One derived silence interval.
    """
    start_time: float
    end_time: float


def save_speaker_label(*, recording: Recording, speaker_id: str, display_name: str) -> SpeakerLabel:
    """
    Create or update the display label for one recording speaker.
    :param recording: Recording owning the speaker.
    :param speaker_id: Canonical speaker identifier produced by diarization.
    :param display_name: User-facing display label for the speaker.
    :return: Persisted SpeakerLabel row.
    :raises ValueError: If any value is blank or the speaker is not present in this recording.
    """
    normalized_speaker_id = (speaker_id or "").strip()
    normalized_display_name = (display_name or "").strip()
    if not normalized_speaker_id:
        raise ValueError("speaker_id must not be empty")
    if not normalized_display_name:
        raise ValueError("display_name must not be empty")
    if not SpeakerSegment.objects.filter(recording=recording, speaker_id=normalized_speaker_id).exists():
        raise ValueError(f"Speaker {normalized_speaker_id} was not found for recording {recording.id}")

    logger.info(
        "speaker_label_save_started",
        extra={
            "recording_id": str(recording.id),
            "speaker_id": normalized_speaker_id,
            "display_name": normalized_display_name,
        }
    )

    with transaction.atomic():
        speaker_label, _created = SpeakerLabel.objects.update_or_create(
            recording=recording,
            speaker_id=normalized_speaker_id,
            defaults={
                "display_name": normalized_display_name,
            },
        )

    logger.info(
        "speaker_label_save_completed",
        extra={
            "recording_id": str(recording.id),
            "speaker_id": normalized_speaker_id,
            "display_name": speaker_label.display_name,
        }
    )
    return speaker_label


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

def derive_silence_intervals(*, speech_segments: list[VadSpeechSegment], duration_milliseconds: int) -> list[SilenceInterval]:
    """
    Derive silence intervals from speech segments and total recording duration.
    :param speech_segments: Speech intervals sorted or unsorted.
    :param duration_milliseconds: Total duration of the recording in milliseconds.
    :return: Silence intervals in integer milliseconds.
    :raises: ValueError: if duration is not positive.
    """
    if duration_milliseconds <= 0:
        raise ValueError('duration_milliseconds must be greater than 0')
    if not speech_segments:
        return [SilenceInterval(start_time=0, end_time=duration_milliseconds)]
    ordered = sorted(speech_segments, key=lambda s: (s.start_time, s.end_time))
    silences: list[SilenceInterval] = []
    cursor = 0
    for segment in ordered:
        if segment.start_time > cursor:
            silences.append(
                SilenceInterval(
                    start_time=cursor,
                    end_time=segment.start_time,
                )
            )
        cursor = max(cursor, segment.end_time)
    if cursor < duration_milliseconds:
        silences.append(
            SilenceInterval(
                start_time=cursor,
                end_time=duration_milliseconds,
            )
        )
    return [silence for silence in silences if silence.end_time > silence.start_time]

def run_vad(*, recording: Recording) -> list[SilenceSegment]:
    """
    Run VAD for a normalized recording and persist derived silence segments.
    :param recording: Recording instance to analyze
    :return: Persisted silence segments
    :raises: ValueError: if recording is missing a normalized file path or duration
    :raises: FileNotFoundError: if the normalized file does not exist.
    :raises: VadError: if the VAD backend fails.
    """
    if not recording.normalized_file_path or not recording.normalized_file_path.strip():
        raise ValueError('recording.normalized_file_path must not be empty.')
    if recording.duration_milliseconds <= 0:
        raise ValueError('recording.duration_milliseconds must be grater than 0')
    logger.info(
        "recording_vad_started",
        extra={
            "recording_id": str(recording.id),
            "normalized_file_path": recording.normalized_file_path,
            "duration_milliseconds": recording.duration_milliseconds,
        }
    )

    model_used = get_vad_model_name()
    speech_segments = run_vad_backend(audio_path=Path(recording.normalized_file_path))
    silence_intervals = derive_silence_intervals(
        speech_segments=speech_segments,
        duration_milliseconds=recording.duration_milliseconds,
    )
    with transaction.atomic():
        SilenceSegment.objects.filter(recording=recording).delete()
        created_segments: list[SilenceSegment] = []
        for interval in silence_intervals:
            created_segments.append(
                SilenceSegment.objects.create(
                    recording=recording,
                    start_time=interval.start_time,
                    end_time=interval.end_time,
                    model_used=model_used,
                )
            )
    logger.info(
        "recording_vad_completed",
        extra={
            "recording_id": str(recording.id),
            "silence_segment_count": len(created_segments),
            "model_used": model_used,
        }
    )
    return created_segments
