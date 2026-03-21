from dataclasses import dataclass
from enum import StrEnum


class BackendName(StrEnum):
    """
    Supported transcription backend identifiers
    """
    MLX_WHISPER = "mlx-whisper"
    WHISPER_CPP = "whisper.cpp"


class ModelName(StrEnum):
    """
    Supported transcription model identifiers
    """
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"


@dataclass(frozen=True)
class ModelSpec:
    """
    Description of a supported transcription model.

    :arg name: Stable model identifier exposed to the application.
    :arg display_name: Human-friendly name for UI or diagnostics.
    :arg description: Short explanation of the model tradeoffs.
    """
    name: ModelName
    display_name: str
    description: str


@dataclass(frozen=True)
class BackendSpec:
    """
    Description of a supported transcription backend.
    :arg name: Stable backend identifier.
    :arg display_name: Human-friendly backend name.
    :arg description: Short explanation of the backend purposes.
    :arg supported_models: Supported model names for this backend.
    :arg is_default_on_macos_apple_silicon: Whether this backend is the preferred default on Apple Silicon macOS.
    :arg is_default_on_linux: Whether this backend is the preferred default on Linux.
    """
    name: BackendName
    display_name: str
    description: str
    supported_models: tuple[ModelName, ...]
    is_default_on_macos_apple_silicon: bool = False
    is_default_on_linux: bool = False
