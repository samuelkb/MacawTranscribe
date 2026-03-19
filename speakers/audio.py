import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from django.conf import settings

logger: Final[logging.Logger] = logging.getLogger(__name__)

class DiarizationError(RuntimeError):
    """Raised when speaker diarization fails."""


@dataclass(frozen=True)
class DiarizationSegment:
    """
    One diarization segment produced by the diarization backend
    """
    speaker_id: str
    start_time: int
    end_time: int


def get_diarization_model_name()-> str:
    """
    Resolve the configured diarization model name
    :return: The diarization model identifier
    """
    return getattr(
        settings,
        "DIARIZATION_MODEL_NAME",
        "pyannote/speaker-diarization-community-1"
    )

def run_pyannote_diarization(*, audio_path: Path) -> list[DiarizationSegment]:
    """
    Run full-recording speaker diarization using pyannote.
    :param audio_path: Path to the normalized audio file
    :return: Diarization segments in integer milliseconds
    :raises ValueError: If ``audio_path`` is empty.
    :raises FileNotFoundError: If ``audio_path`` does not exist.
    :raises DiarizationError: If the pipeline cannot be loaded or execution fails.
    """
    raw_path: str = str(audio_path)
    if not raw_path or not raw_path.strip() or raw_path == ".":
        raise ValueError("audio_path must not be empty")
    if not audio_path.exists():
        raise FileNotFoundError(f"normalized audio file was not found: {audio_path}")
    model_name: str = get_diarization_model_name()
    token = getattr(settings, "HUGGINGFACE_ACCESS_TOKEN", None)
    if token is None:
        logging.warning("HUGGINGFACE_ACCESS_TOKEN not set")
    logger.info(
        "diarization_backend_started",
        extra={
            "audio_path": str(audio_path),
            "model_used": model_name,
        }
    )
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise DiarizationError("pyannote.audio is not available") from exc

    try:
        pipeline = Pipeline.from_pretrained(
            model_name,
            token=token,
        )
    except Exception as exc:
        raise DiarizationError(f"failed to load diarization pipeline: {model_name}") from exc

    try:
        diarization = pipeline(str(audio_path))
    except Exception as exc:
        raise DiarizationError("diarization pipeline execution failed") from exc

    segments: list[DiarizationSegment] = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        start_time, end_time = int(turn.start * 1000), int(turn.end * 1000)
        if end_time <= start_time:
            continue
        segments.append(
            DiarizationSegment(
                speaker_id=str(speaker_label),
                start_time=start_time,
                end_time=end_time,
            )
        )
    if not segments:
        raise DiarizationError("diarization returned no speaker segments")
    logger.info(
        "diarization_backend_completed",
        extra={
            "audio_path": str(audio_path),
            "model_used": model_name,
            "segments_count": len(segments),
        }
    )
    return segments