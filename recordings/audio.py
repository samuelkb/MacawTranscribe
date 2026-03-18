import logging
import subprocess
from typing import Final

from pathlib import Path

logger: Final[logging.Logger] = logging.getLogger("app.recordings")

class AudioNormalizationError(RuntimeError):
    """
    Raised when audio normalization fails.
    """


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
