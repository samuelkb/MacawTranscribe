import logging
import subprocess
import tempfile
from typing import Final

from pathlib import Path

from recordings.models import Chunk

logger: Final[logging.Logger] = logging.getLogger(__name__)

class AudioNormalizationError(RuntimeError):
    """
    Raised when audio normalization fails.
    """


class AudioProbeError(RuntimeError):
    """
    Raised when audio metadata probing fails.
    """


class ChunkAudioExtractionError(RuntimeError):
    """Raised when chunk audio extraction fails."""


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

def extract_chunk_audio(*, chunk: Chunk) -> Path:
    """
    Extract a chunk audio file from a recordings normalized audio.

    The extracted chunk is written as a temporary WAV file and is intended to be consumed immediately by a transcription
    code.
    :param chunk: Chunk object instance to extract audio from.
    :return: Path to the extracted temporary chunk WAV file.
    :raises ValueError: If the chunk timing is invalid or normalized audio path is missing.
    :raises FileNotFoundError: If the normalized audio file does not exist.
    :raises ChunkAudioError: If ffmpeg execution fails.
    """
    recording = chunk.recording

    if not recording.normalized_file_path or not recording.normalized_file_path.strip():
        raise ValueError("recording.normalized_file_path must not be empty")
    if chunk.start_time < 0:
        raise ValueError("chunk.start_time must not be negative")
    if chunk.end_time <= chunk.start_time:
        raise ValueError("chunk.end_time must be greater than chunk.start_ms")
    source_path = Path(recording.normalized_file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"normalized audio file was not found: {source_path}")

    start_seconds = chunk.start_time / 1000
    duration_seconds = (chunk.end_time - chunk.start_time) / 1000
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix=f"chunk-{chunk.id}-",
        delete=False,
    )
    output_path = Path(temp_file.name)
    temp_file.close()
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        str(source_path),
        "-t",
        str(duration_seconds),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info(
        "chunk_audio_extraction_started",
        extra={
            "recording_id": str(recording.id),
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
        },
    )
    try:
        logger.debug(f"ffmpeg executing command: {command}")
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        output_path.unlink(missing_ok=True)
        raise ChunkAudioExtractionError("ffmpeg executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise ChunkAudioExtractionError(f"ffmpeg chunk extraction failed: {stderr or "unknown error"}") from exc
    logger.info(
        "chunk_audio_extraction_completed",
        extra={
            "recording_id": str(recording.id),
            "chunk_id": str(chunk.id),
            "chunk_index": chunk.chunk_index,
            "output_path": str(output_path),
        }
    )
    return output_path