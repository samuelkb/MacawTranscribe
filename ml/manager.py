import logging
import platform
from dataclasses import dataclass
from typing import Final

from ml.backends.base import TranscriptionBackend, LoadedModelHandle
from ml.backends.mlx_whisper_backend import MlxWhisperBackend
from ml.registry import list_supported_backends, list_supported_models, SUPPORTED_BACKENDS, SUPPORTED_MODELS, \
    backend_supports_model
from ml.types import BackendName, ModelName, BackendSpec, ModelSpec

logger: Final[logging.Logger] = logging.getLogger(__name__)

class UnsupportedBackendError(ValueError):
    """Raised when a backend is not supported."""


class UnsupportedModelError(ValueError):
    """Raised when a model is not supported."""


@dataclass(frozen=True)
class ResolvedModelSelection:
    """
    Fully resolved backend/model selection.
    """

    backend: BackendName
    model: ModelName


class ModelManager:
    """
    High-level runtime manager for transcription backends and models.
    """

    def list_backends(self) -> list[BackendSpec]:
        logger.info("Running list_backends")
        return list_supported_backends()

    def list_models(self) -> list[ModelSpec]:
        logger.info("Running list_models")
        return list_supported_models()

    def get_default_backend(self) -> BackendName:
        logger.info("Running get_default_backend")
        system = platform.system().lower()
        logger.debug(f"system: {system}")
        machine = platform.machine().lower()
        logger.debug(f"machine: {machine}")
        is_macos = system == "darwin"
        is_apple_silicon = machine in {"arm64", "aarch64"}
        is_linux = system == "linux"

        if is_macos and is_apple_silicon:
            for spec in SUPPORTED_BACKENDS.values():
                if spec.is_default_on_macos_apple_silicon:
                    return spec.name
        if is_linux:
            for spec in SUPPORTED_BACKENDS.values():
                if spec.is_default_on_linux:
                    return spec.name

        return BackendName.MLX_WHISPER

    def get_default_model(self) -> ModelName:
        logger.info("Running get_default_model")
        return ModelName.MEDIUM

    def resolve_selection(self, *, backend: BackendName | None = None, model: ModelName | None = None) -> ResolvedModelSelection:
        """
        :param backend: Backend name
        :param model: Model name
        :return: ResolvedModelSelection instance
        """
        logger.info("Running resolve_selection")
        resolved_backend: BackendName | None = backend or self.get_default_backend()
        logger.debug(f"resolved_backend: {resolved_backend}")
        resolved_model: ModelName | None = model or self.get_default_model()
        logger.debug(f"resolved_model: {resolved_model}")

        if resolved_backend not in SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(f"Unsupported backend {resolved_backend}")
        if resolved_model not in SUPPORTED_MODELS:
            raise UnsupportedModelError(f"Unsupported model {resolved_model}")
        if not backend_supports_model(
            backend=resolved_backend,
            model=resolved_model,
        ):
            raise UnsupportedModelError(f"Backend {resolved_backend.value} does not support {resolved_model.value}")
        logger.info(
            "resolve_selection_completed",
            extra={"backend": resolved_backend.value, "model": resolved_model.value},
        )
        return ResolvedModelSelection(
            backend=resolved_backend,
            model=resolved_model,
        )

    def get_backend(self, *, backend: BackendName) -> TranscriptionBackend:
        logger.info("Running get_backend")
        if backend == BackendName.MLX_WHISPER:
            return MlxWhisperBackend()
        if backend == BackendName.WHISPER_CPP:
            raise NotImplementedError("whisper.cpp backend not implemented yet")
        logger.exception(f"backend {backend} not supported")
        raise UnsupportedBackendError(f"Unsupported backend {backend}")

    def is_model_available(self, *, backend: BackendName | None = None, model: ModelName | None = None) -> bool:
        logger.info("Running is_model_available")
        selection = self.resolve_selection(
            backend=backend,
            model=model,
        )
        backend_implementation = self.get_backend(backend=selection.backend)
        logger.debug(f"backend_implementation: {backend_implementation}")
        return backend_implementation.is_model_available(model=selection.model)

    def ensure_model_available(self, *, backend: BackendName | None = None, model: ModelName | None = None) -> ResolvedModelSelection:
        selection = self.resolve_selection(
            backend=backend,
            model=model,
        )
        logger.debug(f"selection: {selection}")
        backend_implementation = self.get_backend(backend=selection.backend)
        logger.debug(f"backend_implementation: {backend_implementation}")
        logger.info(
            "model_ensure_started",
            extra={
                "backend": selection.backend.value,
                "model": selection.model.value,
            }
        )
        backend_implementation.ensure_model_available(model=selection.model)
        logger.info(
            "model_ensure_completed",
            extra={
                "backend": selection.backend.value,
                "model": selection.model.value,
            }
        )
        return selection

    def load_model(self, * , backend: BackendName | None = None, model: ModelName | None = None) -> tuple[ResolvedModelSelection, TranscriptionBackend, LoadedModelHandle]:
        logger.info(
            "Running load_model",
            extra={
                "backend": backend.value,
                "model": model.value,
            }
        )
        selection = self.ensure_model_available(
            backend=backend,
            model=model,
        )
        backend_implementation = self.get_backend(backend=selection.backend)
        loaded_model = backend_implementation.load_model(model=selection.model)
        logger.info(
            "load_model_completed",
            extra={
                "backend": selection.backend.value,
                "model": selection.model.value,
            }
        )
        return selection, backend_implementation, loaded_model