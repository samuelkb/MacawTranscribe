import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from ml.types import ModelName, BackendName

logger: Final[logging.Logger] = logging.getLogger(__name__)


HeartbeatCallback: TypeAlias = Callable[[], None]


class ModelAvailabilityError(RuntimeError):
    """Raised when model availability cannot be determined."""


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be downloaded or prepared."""


class ModelLoadError(RuntimeError):
    """Raised when a backend cannot load a model in memory."""


class TranscriptionBackendError(RuntimeError):
    """Raised when a transcription execution fails."""


@dataclass(frozen=True)
class TranscribedWord:
    """
    Word-level transcription output returned by a backend.
    """
    word_index: int
    text: str
    start_time: int
    end_time: int
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """
    Result of transcribing one chunk audio file.
    """
    full_text: str
    words: tuple[TranscribedWord, ...]
    model_used: ModelName
    backend_used: BackendName


class LoadedModelHandle(ABC):
    """
    Abstract base class for an in-memory loaded model.

    Concrete backends may wrap any runtime-specific object here.
    """

    @property
    @abstractmethod
    def backend_name(self) -> BackendName:
        """
        Backend that created this loaded model handle.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> ModelName:
        """
        Model loaded in this handle.
        """
        raise NotImplementedError


class TranscriptionBackend(ABC):
    """
    Abstract contract implemented by all transcription backends.

    The worker should depend on this interface rather than on backend-specific details internals.
    """

    @property
    @abstractmethod
    def name(self) -> BackendName:
        """
        Stable backend identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def supports_model(self, *, model: ModelName) -> bool:
        """
        Return whether this backend supports the given model.
        """
        raise NotImplementedError

    @abstractmethod
    def is_model_available(self, *, model: ModelName) -> bool:
        """
        :return: Whether this backend supports the given model.
        :raises ModelAvailabilityError: If availability cannot be determined.
        """
        raise NotImplementedError

    @abstractmethod
    def ensure_model_available(self, *, model: ModelName) -> None:
        """
        Ensure the requested model is available locally. This may download, convert or prepare model artifacts.
        :raises ModelDownloadError: If model cannot be prepared.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *, model: ModelName) -> LoadedModelHandle:
        """
        Load a model into memory and return a backend-specific handle.
        :raises ModelLoadError: If model cannot be loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe(
            self,
            *,
            loaded_model: LoadedModelHandle,
            audio_path: Path,
            heartbeat_callback: HeartbeatCallback | None = None
    ) -> TranscriptionResult:
        """
        Transcribe one audio file using a previously loaded model.
        :param loaded_model: A handle returned by ``load_model``.
        :param audio_path: Path to the chunk audio file to transcribe.
        :param heartbeat_callback: A callback that will be called when a transcription occurs.
        :return: Word-level transcription result.
        :raises ValueError: If the input is invalid.
        :raises FileNotFoundError: If the audio file does not exist.
        :raises TranscriptionBackendError: If transcription execution fails.
        """
        raise NotImplementedError