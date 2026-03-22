import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from django.conf import settings

from ml.backends.base import LoadedModelHandle, TranscriptionBackend, ModelAvailabilityError, ModelDownloadError, \
    ModelLoadError, TranscriptionResult, TranscriptionBackendError, TranscribedWord
from ml.registry import backend_supports_model
from ml.types import ModelName, BackendName

logger: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True)
class MlxWhisperLoadedModelHandle(LoadedModelHandle):
    """
    Lightweight handle for an MLX Whisper model.

    The public mlx-whisper docs center on `mlx_whisper.transcribe(...)` with
    `path_or_hf_repo`, rather than on a documented public persistent load API.
    So this handle stores the resolved local model directory and metadata,
    which keeps the worker contract stable while remaining aligned with the
    documented interface. contentReference[oaicite:2]{index=2}
    """
    _model_name: ModelName
    local_model_dir: Path

    @property
    def model_name(self) -> ModelName:
        return self._model_name

    @property
    def backend_name(self) -> BackendName:
        return BackendName.MLX_WHISPER


class MlxWhisperBackend(TranscriptionBackend):
    """
    MLX Whisper backend for Apple Silicon Macs.

    This backend uses pre-converted Whisper checkpoints from the MLX Community,
    which the official docs describe as available on Hugging Face, and the
    documented Python API transcribes via `mlx_whisper.transcribe(...)` with
    `path_or_hf_repo`. contentReference[oaicite:3]{index=3}
    """
    MODEL_REPO_MAP: dict[ModelName, str] = {
        ModelName.SMALL: "mlx-community/whisper-small",
        ModelName.MEDIUM: "mlx-community/whisper-medium",
        ModelName.LARGE_V3: "mlx-community/whisper-large-v3-turbo",
    }
    REQUIRED_MODEL_FILES: tuple[str, ...] = (
        "config.json",
        "weights.npz"
    )

    @property
    def name(self) -> BackendName:
        return BackendName.MLX_WHISPER

    def supports_model(self, *, model: ModelName) -> bool:
        logger.info(f"Running supports_model on {model}")
        return backend_supports_model(
            backend=BackendName.MLX_WHISPER,
            model=model,
        )

    def is_model_available(self, *, model: ModelName) -> bool:
        logger.info(f"Running is_model_available on {model}")
        if not self.supports_model(model=model):
            raise ModelAvailabilityError(f"Backend {self.name.value} does not support model {model}")
        model_dir = self.get_local_model_dir(model=model)
        return all((model_dir/filename).exists() for filename in self.REQUIRED_MODEL_FILES)

    def ensure_model_available(self, *, model: ModelName) -> None:
        logger.info(f"Running ensure_model_available on {model}")
        if not self.supports_model(model=model):
            raise ModelDownloadError(f"Backend {self.name.value} does not support model {model}")
        if self.is_model_available(model=model):
            return
        repo_id = self.get_hf_repo_for_model(model=model)
        local_dir = self.get_local_model_dir(model=model)
        local_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "mlx_whisper_model_download_started",
            extra={
                "backend": self.name.value,
                "model": model.value,
                "repo_id": repo_id,
                "local_dir": str(local_dir),
            }
        )
        try:
            logger.info("Importing snapshot_download from huggingface_hub")
            from huggingface_hub import snapshot_download
        except Exception as exc:
            logger.exception("Import snapshot_download failed")
            raise ModelDownloadError("huggingface_hub is required to download MLX Whisper models") from exc

        allow_patterns = list(self.REQUIRED_MODEL_FILES) + [
            "tokenizer*",
            "*.tiktoken",
            "preprocessor_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocabulary.json",
            "merges.txt",
        ]

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
                token=getattr(settings, "HUGGINGFACE_ACCESS_TOKEN", None)
            )
        except Exception as exc:
            logger.exception("Snapshot download failed")
            raise ModelDownloadError(f"Failed to download MLX Whisper model {model.value} from {repo_id}") from exc
        if not self.is_model_available(model=model):
            raise ModelDownloadError(f"Downloaded MLX Whisper model {model.value}, but required files are missing")
        logger.info(
            "mlx_whisper_model_download_completed",
            extra={
                "backend": self.name.value,
                "model": model.value,
                "repo_id": repo_id,
                "local_dir": str(local_dir),
            }
        )

    def load_model(self, *, model: ModelName) -> LoadedModelHandle:
        logger.info(f"Running load_model on {model.value}")
        if not self.supports_model(model=model):
            raise ModelLoadError(f"Backend {self.name.value} does not support model {model.value}")
        self.ensure_model_available(model=model)
        local_dir = self.get_local_model_dir(model=model)
        if not local_dir.exists():
            raise ModelLoadError(f"Local MLX Whisper model directory does not exist: {local_dir}")
        logger.info(
            "mlx_whisper_model_loaded",
            extra={
                "backend": self.name.value,
                "model": model.value,
                "local_dir": str(local_dir),
            },
        )
        return MlxWhisperLoadedModelHandle(
            _model_name=model,
            local_model_dir=local_dir,
        )

    def transcribe(self, *, loaded_model: LoadedModelHandle, audio_path: Path) -> TranscriptionResult:
        raw_path = str(audio_path)
        if not raw_path or not raw_path.strip() or raw_path == ".":
            raise ValueError("audio_path must not be empty")
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file was not found: {audio_path}")
        if not isinstance(loaded_model, MlxWhisperLoadedModelHandle):
            raise ValueError("loaded_model must be an MlxWhisperLoadedModelHandle")
        try:
            logger.info("importing mlx_whisper")
            import mlx_whisper
        except Exception as exc:
            logger.exception("mlx_whisper import failed")
            raise TranscriptionBackendError("mlx_whisper is not installed") from exc
        logger.info(
            "mlx_whisper_transcription_started",
            extra={
                "backend": self.name.value,
                "model": loaded_model.model_name.value,
                "audio_path": str(audio_path),
                "local_model_dir": str(loaded_model.local_model_dir),
            },
        )
        try:
            logger.info(f"Running mlx_whisper.transcribe on {audio_path}")
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=str(loaded_model.local_model_dir),
                word_timestamps=True,
            )
        except Exception as exc:
            logger.exception("mlx_whisper.transcribe failed")
            raise TranscriptionBackendError("MLX Whisper transcription failed") from exc

        full_text = (result.get("text") or "").strip()
        logger.debug(f"full_text: {full_text}")
        segments = result.get("segments") or []
        logger.debug(f"segments: {segments}")
        words: list[TranscribedWord] = []
        word_index = 0
        for segment in segments:
            segment_words = segment.get("words") or []
            for item in segment_words:
                text = (item.get("word") or "").strip()
                start_s = item.get("start")
                end_s = item.get("end")
                confidence = item.get("probability")

                if not text:
                    continue
                if start_s is None or end_s is None:
                    continue

                start_time = int(float(start_s) * 1000)
                end_time = int(float(end_s) * 1000)
                if end_time <= start_time:
                    continue

                words.append(
                    TranscribedWord(
                        word_index=word_index,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=float(confidence) if confidence is not None else None,
                    )
                )
                word_index += 1
        logger.info(
            "mlx_whisper_transcription_completed",
            extra={
                "backend": self.name.value,
                "model": loaded_model.model_name.value,
                "audio_path": str(audio_path),
                "word_count": len(words),
            }
        )
        return TranscriptionResult(
            full_text=full_text,
            words=tuple(words),
            model_used=loaded_model.model_name,
            backend_used=self.name,
        )

    def get_hf_repo_for_model(self, *, model: ModelName) -> str:
        logger.info(f"get_hf_repo_for_model on {model.value}")
        try:
            return self.MODEL_REPO_MAP[model]
        except KeyError as exc:
            logger.exception(f"get_hf_repo_for_model failed for {model.value}")
            raise ModelAvailabilityError(f"No Hugging Face repo mapping registered for model {model.value}")

    def get_models_base_dir(self) -> Path:
        logger.info("get_models_base_dir")
        configured = getattr(settings, "MODELS_BASE_DIR", "models")
        return Path(configured)

    def get_local_model_dir(self, *, model: ModelName) -> Path:
        logger.info(f"get_local_model_dir on {model.value}")
        return self.get_models_base_dir() / "mlx" / model.value
