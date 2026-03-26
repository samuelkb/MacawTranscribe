from dataclasses import dataclass

from ml.backends.base import TranscriptionBackend, LoadedModelHandle
from ml.types import BackendName, ModelName


@dataclass(frozen=True, slots=True)
class LoadedWorkerRuntime:
    """
    Loaded transcription runtime owned by a single worker process.

    A worker process loads exactly one backend/model runtime at boot and reuses it for all compatible jobs during
    the process lifetime.
    """
    backend: BackendName
    model: ModelName
    backend_impl: TranscriptionBackend
    loaded_model: LoadedModelHandle

    @property
    def partition_key(self) -> str:
        return f"{self.backend.value}:{self.model.value}"

