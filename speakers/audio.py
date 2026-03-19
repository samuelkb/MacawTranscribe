import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, runtime_checkable, Protocol, Any

import torch
from django.conf import settings
from pyannote.audio.pipelines.utils.hook import ProgressHook, Hooks, TimingHook, ArtifactHook

logger: Final[logging.Logger] = logging.getLogger(__name__)

logging.getLogger("pyannote").setLevel(logging.DEBUG)
logging.getLogger("pyannote.audio").setLevel(logging.DEBUG)

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


@runtime_checkable
class SupportsIterTracks(Protocol):
    """
    Structural type for pyannote annotation-like objects.

    Any object implementing `itertracks` with a compatible signature can be treated as an annotation for
    downstream segment extraction.
    """

    def itertracks(self,yield_label: bool = False) -> Any:
        """
        Iterate over annotation tracks.
        :param yield_label: Whether to yield speaker/track labels together with each track
        :return: An iterator provided by the underlying pyannote object
        """
        ...


class DjangoLoggingHook:
    """
    Emit detailed logs for every pyannote pipeline step. Compatible with pyannote hook signature:
        hook(step_name, step_artifact, file=None, total=None, completed=None)
    """
    def __enter__(self) -> "DjangoLoggingHook":
        self.__step_started_at = {}
        self.__last_logged_progress = {}
        self.__pipeline_started_at = time.perf_counter()
        logger.info("django_logging_hook_enter")
        return self

    def __exit__(self, exc_type: str, exc_val: str, exc_tb: str) -> None:
        elapsed = time.perf_counter() - self.__pipeline_started_at
        if exc_val:
            logger.exception(
                "pyannote_pipeline_failed",
                extra={
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
        else:
            logger.info(
                "pyannote_pipeline_completed",
                extra={"elapsed_seconds": round(elapsed, 3)},
            )
    def __call__(self, step_name: str, step_artifact: ArtifactHook, file: Path = None, total: int = None, completed: int = None) -> None:
        if completed is not None and total is not None:
            if completed == 0 and step_name not in self.__step_started_at:
                self.__step_started_at[step_name] = time.perf_counter()
                logger.info(
                    "pyannote_step_started",
                    extra={"step_name": step_name, "total": total },
                )
            percent = 100 if total == 0 else int((completed / total) * 100)
            prev = self.__last_logged_progress.get(step_name, None)
            if prev != percent:
                self.__last_logged_progress[step_name] = percent
                logger.info(
                    "pyannote_step_progress",
                    extra={
                        "step_name": step_name,
                        "completed": completed,
                        "total": total,
                        "percent": percent,
                    },
                )
            if completed >= total:
                started = self.__step_started_at.get(step_name)
                elapsed = None
                if started is not None:
                    elapsed = round(time.perf_counter() - started, 3)

                logger.info(
                    "pyannote_step_finished",
                    extra={
                        "step_name": step_name,
                        "total": total,
                        "elapsed_seconds": elapsed,
                        "artifact_type": (
                            type(step_artifact).__name__
                            if step_artifact is not None
                            else None
                        ),
                    },
                )
            else:
                logger.info(
                    "pyannote_step_event",
                    extra={
                        "step_name": step_name,
                        "artifact_type": (
                            type(step_artifact).__name__
                            if step_artifact is not None
                            else None
                        ),
                    },
                )


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

def get_annotation(diarization_result: object) -> SupportsIterTracks:
    """
    Extract an annotation-like object from a diarization pipeline result

    Some pyannote pipeline versions return the annotation directly, while others return a wrapper object such as
     ``DiarizeOutput`` containing the actual annotation under a known attribute. This function normalizes both
    returning the first object that exposes an ``itertracks`` method.
    :param diarization_result: Raw object returned by the diarization pipeline.
    :return: An annotation-like object supporting ``itertracks``
    :raises TypeError: If no annotation-like object can be found in the pipeline result.
    """
    candidate_attributes: tuple[str, ...] = (
        "speaker_diarization",
        "diarization",
        "annotation",
    )
    for attribute_name in candidate_attributes:
        candidate: object | None = getattr(diarization_result, attribute_name, None)
        if isinstance(candidate, SupportsIterTracks):
            return candidate
    if isinstance(diarization_result, SupportsIterTracks):
        return diarization_result

    available_attributes: list[str] = [
        attribute_name for attribute_name in dir(diarization_result) if not attribute_name.startswith("_")
    ]
    raise TypeError(
        "Unsupported_diarization_result_type",
        f"{type(diarization_result)!r}",
        "Could not find an annotation-like object exposing 'itertracks'",
        f"Available public attributes: {available_attributes}",
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
        pipeline.to(device)
        logger.info(
            "torch_device_debug",
            extra={
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
            },
        )
    except Exception as exc:
        raise DiarizationError(f"failed to load diarization pipeline: {model_name}") from exc

    try:
        with Hooks(
            ProgressHook(transient=False),
            TimingHook(file_key="timing"),
            ArtifactHook(file_key="artifacts"),
            DjangoLoggingHook()
        ) as hook:
            diarization = pipeline(str(audio_path), hook=hook)
    except Exception as exc:
        raise DiarizationError("diarization pipeline execution failed") from exc

    annotation: SupportsIterTracks = get_annotation(diarization)
    segments: list[DiarizationSegment] = []
    for turn, _, speaker_label in annotation.itertracks(yield_label=True):
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