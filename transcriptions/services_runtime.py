import logging
from typing import Final

from ml.manager import ModelManager
from ml.types import BackendName, ModelName
from transcriptions.runtime import LoadedWorkerRuntime

logger: Final[logging.Logger] = logging.getLogger(__name__)


def load_worker_transcription_runtime(*, backend: BackendName | None = None, model: ModelName | None = None) -> LoadedWorkerRuntime:
    """
    Load the transcription runtime for a worker process.

    This resolves the effective backend/model selection through ModelManager, loads the model once, and returns a typed
    runtime object that can be reused across many chunk transcription jobs in the same worker process.
    :param backend: Optional requested backend
    :param model: Optional requested model
    :return: Loaded runtime owned by a single worker process.
    """
    logger.info(
        "worker_transcription_runtime_loading",
        extra={
            "backend": backend.value if backend else None,
            "model": model.value if model else None,
        }
    )
    manager = ModelManager()
    selection, backend_impl, loaded_model = manager.load_model(backend=backend, model=model)
    runtime = LoadedWorkerRuntime(
        backend=selection.backend,
        model=selection.model,
        backend_impl=backend_impl,
        loaded_model=loaded_model,
    )
    logger.info(
        "worker_transcription_runtime_loaded",
        extra={
            "backend": runtime.backend.value,
            "model": runtime.model.value,
            "partition_key": runtime.partition_key,
            "backend_impl_class": type(runtime.backend_impl).__name__,
            "loaded_model_class": type(runtime.loaded_model).__name__,
        }
    )
    return runtime