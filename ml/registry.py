import logging
from typing import Final

from ml.types import ModelName, ModelSpec, BackendName, BackendSpec

logger: Final[logging.Logger] = logging.getLogger(__name__)

SUPPORTED_MODELS: dict[ModelName, ModelSpec] = {
    ModelName.SMALL: ModelSpec(
        name=ModelName.SMALL,
        display_name="Small",
        description="Fast draft transcription with lower accuracy."
    ),
    ModelName.MEDIUM: ModelSpec(
        name=ModelName.MEDIUM,
        display_name="Medium",
        description="Balanced accuracy and speed."
    ),
    ModelName.LARGE_V3: ModelSpec(
        name=ModelName.LARGE_V3,
        display_name="Large v3",
        description="Highest accuracy with higher resoruce usage."
    ),
}

SUPPORTED_BACKENDS: dict[BackendName, BackendSpec] = {
    BackendName.MLX_WHISPER: BackendSpec(
        name=BackendName.MLX_WHISPER,
        display_name="MLX Whisper",
        description="Apple Silicon optimized Whisper runtime.",
        supported_models=(
            ModelName.SMALL,
            ModelName.MEDIUM,
            ModelName.LARGE_V3,
        ),
        is_default_on_macos_apple_silicon=True,
        is_default_on_linux=False,
    ),
    BackendName.WHISPER_CPP: BackendSpec(
        name=BackendName.WHISPER_CPP,
        display_name="whisper.cpp",
        description="Portable Whisper backend for cross-platform use.",
        supported_models=(
            ModelName.SMALL,
            ModelName.MEDIUM,
            ModelName.LARGE_V3,
        ),
        is_default_on_macos_apple_silicon=False,
        is_default_on_linux=True,
    )
}

def list_supported_backends() -> list[BackendSpec]:
    """
    :return: list[BackendSpec]: supported backends in registry order.
    """
    logger.info("executing list_supported_backends")
    return list(SUPPORTED_BACKENDS.values())

def list_supported_models() -> list[ModelSpec]:
    """
    :return: list[ModelSpec]: supported models in registry order.
    """
    logger.info("executing list_supported_models")
    return list(SUPPORTED_MODELS.values())

def get_backend_spec(*, backend: BackendName) -> BackendSpec:
    """
    :param backend: BackendName
    :return: The spec for a backend.
    :raises KeyError: if backend is not registered.
    """
    logger.info(f"executing get_backend_spec for {backend}")
    return SUPPORTED_BACKENDS[backend]

def get_model_spec(*, model: ModelName) -> ModelSpec:
    """
    :param model: ModelName
    :return: The spec for a model.
    :raises KeyError: if model is not registered.
    """
    logger.info(f"executing get_model_spec for {model}")
    return SUPPORTED_MODELS[model]

def backend_supports_model(*, backend: BackendName, model: ModelName) -> bool:
    """
    :param backend: BackendName
    :param model: ModelName
    :return: True if backend supports model, False otherwise.
    """
    logger.info(f"executing backend_supports_model for {backend}, {model}")
    backend_spec = get_backend_spec(backend=backend)
    return model in backend_spec.supported_models
