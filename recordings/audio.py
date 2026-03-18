import logging
import subprocess
from typing import Final

from pathlib import Path

logger: Final[logging.Logger] = logging.getLogger(__name__)

class AudioNormalizationError(RuntimeError):
    """
    Raised when audio normalization fails.
    """


class AudioProbeError(RuntimeError):
    """
    Raised when audio metadata probing fails.
    """

def probe_audio_duration_milliseconds(*, input_path: Path) -> int:
    """
    Probe the duration of an audio file using ffmpeg.

    The returned duration is expressed in integer milliseconds, and it is intended to be the canonical timing unit.
    :param input_path: Path to the audio file to inspect.
    :return: int: Duration in integer milliseconds.
    :raises ValueError: If input_path does not exist
    :raises FileNotFoundError: If the target file does not exist
    :raises AudioProbeError: If ffprobe is missing, fails, or returns an invalid duration.
    """
    raw_path = str(input_path)
    if not raw_path or not raw_path.strip() or raw_path == ".":
        raise ValueError("input_path must not be empty")
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file was not found: {input_path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]

    logger.info(
        "audio_probe_started",
        extra={
            "input_path": str(input_path),
        },
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AudioProbeError("ffprobe executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise AudioProbeError(
            f"ffprobe failed: {stderr or 'unknown error'}"
        ) from exc

    raw_duration = result.stdout.strip()
    if not raw_duration:
        raise AudioProbeError("ffprobe returned an empty duration")

    try:
        duration_seconds = float(raw_duration)
    except ValueError as exc:
        raise AudioProbeError(
            f"ffprobe returned a non-numeric duration: {raw_duration!r}"
        ) from exc

    duration_milliseconds = int(duration_seconds * 1000)
    if duration_milliseconds <= 0:
        raise AudioProbeError(
            f"ffprobe returned a non-positive duration: {duration_seconds}"
        )

    logger.info(
        "audio_probe_completed",
        extra={
            "input_path": str(input_path),
            "duration_milliseconds": duration_milliseconds,
        },
    )

    return duration_milliseconds


def _run_ffmpeg_normalization(*, input_path:Path, output_path:Path) -> None:
    """
    Normalize an audio file into a mono 16kHz PCM WAV file using ffmpeg.
    :param input_path: Path to the original uploaded audio file.
    :param output_path: Path where the normalized audio file should be written.
    :return: None
    :raises AudioNormalizationError: If ffmpeg returns a non-zero exit code or cannot be executed.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info(
        "audio_ffmpeg_normalization_started",
        extra={
            "input_path": str(input_path),
        },
    )

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exec:
        raise AudioNormalizationError("ffmpeg executable was not found") from exec
    except subprocess.CalledProcessError as exec:
        stderr = exec.stderr.strip() if exec.stderr else ""
        raise AudioNormalizationError(
            f"ffmpeg normalization failed: {stderr or "unknown error"}"
        ) from exec
