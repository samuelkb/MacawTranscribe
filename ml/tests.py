from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock

from django.test import SimpleTestCase, override_settings

from ml.backends.base import TranscribedWord, TranscriptionResult, ModelAvailabilityError, ModelDownloadError, \
    ModelLoadError, TranscriptionBackendError, LoadedModelHandle, TranscriptionBackend
from ml.backends.mlx_whisper_backend import MlxWhisperBackend, MlxWhisperLoadedModelHandle
from ml.manager import ModelManager, UnsupportedBackendError, ResolvedModelSelection
from ml.registry import list_supported_backends, list_supported_models, get_backend_spec, get_model_spec, \
    backend_supports_model
from ml.types import BackendName, ModelName


class ModelRegistryTests(SimpleTestCase):
    def test_list_supported_backends_returns_expected_backends(self) -> None:
        backends = list_supported_backends()
        backend_names = [backend.name for backend in backends]

        self.assertEqual(
            backend_names,
            [
                BackendName.MLX_WHISPER,
                BackendName.WHISPER_CPP,
            ],
        )

    def test_list_supported_models_returns_expected_models(self) -> None:
        models = list_supported_models()
        model_names = [model.name for model in models]

        self.assertEqual(
            model_names,
            [
                ModelName.SMALL,
                ModelName.MEDIUM,
                ModelName.LARGE_V3,
            ],
        )

    def test_get_backend_spec_returns_expected_spec(self) -> None:
        spec = get_backend_spec(backend=BackendName.MLX_WHISPER)

        self.assertEqual(spec.name, BackendName.MLX_WHISPER)
        self.assertTrue(spec.is_default_on_macos_apple_silicon)
        self.assertFalse(spec.is_default_on_linux)

    def test_get_model_spec_returns_expected_spec(self) -> None:
        spec = get_model_spec(model=ModelName.MEDIUM)

        self.assertEqual(spec.name, ModelName.MEDIUM)
        self.assertEqual(spec.display_name, "Medium")

    def test_backend_supports_model_returns_true_for_supported_combination(self) -> None:
        self.assertTrue(
            backend_supports_model(
                backend=BackendName.MLX_WHISPER,
                model=ModelName.LARGE_V3,
            )
        )

    def test_backend_supports_model_returns_true_for_whisper_cpp(self) -> None:
        self.assertTrue(
            backend_supports_model(
                backend=BackendName.WHISPER_CPP,
                model=ModelName.SMALL,
            )
        )


class BaseBackendContractTests(SimpleTestCase):
    def test_transcribed_word_dataclass_stores_values(self) -> None:
        word = TranscribedWord(
            word_index=0,
            text="hello",
            start_time=100,
            end_time=450,
            confidence=0.98,
        )

        self.assertEqual(word.word_index, 0)
        self.assertEqual(word.text, "hello")
        self.assertEqual(word.start_time, 100)
        self.assertEqual(word.end_time, 450)
        self.assertEqual(word.confidence, 0.98)

    def test_transcription_result_dataclass_stores_values(self) -> None:
        word = TranscribedWord(
            word_index=0,
            text="hello",
            start_time=100,
            end_time=450,
            confidence=0.98,
        )

        result = TranscriptionResult(
            full_text="hello",
            words=(word,),
            model_used=ModelName.MEDIUM,
            backend_used=BackendName.MLX_WHISPER,
        )

        self.assertEqual(result.full_text, "hello")
        self.assertEqual(len(result.words), 1)
        self.assertEqual(result.model_used, ModelName.MEDIUM)
        self.assertEqual(result.backend_used, BackendName.MLX_WHISPER)

    def test_custom_exceptions_are_runtime_errors(self) -> None:
        self.assertTrue(issubclass(ModelAvailabilityError, RuntimeError))
        self.assertTrue(issubclass(ModelDownloadError, RuntimeError))
        self.assertTrue(issubclass(ModelLoadError, RuntimeError))
        self.assertTrue(issubclass(TranscriptionBackendError, RuntimeError))

    def test_loaded_model_handle_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            LoadedModelHandle()

    def test_transcription_backend_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            TranscriptionBackend()


class MlxWhisperBackendTests(SimpleTestCase):
    def setUp(self) -> None:
        self.backend = MlxWhisperBackend()

    def test_name_returns_mlx_whisper(self) -> None:
        self.assertEqual(self.backend.name, BackendName.MLX_WHISPER)

    def test_supports_registered_models(self) -> None:
        self.assertTrue(self.backend.supports_model(model=ModelName.SMALL))
        self.assertTrue(self.backend.supports_model(model=ModelName.MEDIUM))
        self.assertTrue(self.backend.supports_model(model=ModelName.LARGE_V3))

    def test_get_hf_repo_for_model_returns_expected_repo(self) -> None:
        self.assertEqual(
            self.backend.get_hf_repo_for_model(model=ModelName.LARGE_V3),
            "mlx-community/whisper-large-v3",
        )

    def test_is_model_available_returns_true_when_required_files_exist(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_BASE_DIR=tmp_dir):
                model_dir = self.backend.get_local_model_dir(model=ModelName.MEDIUM)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.json").write_text("{}")
                (model_dir / "weights.npz").write_bytes(b"fake")

                self.assertTrue(self.backend.is_model_available(model=ModelName.MEDIUM))

    def test_is_model_available_returns_false_when_required_files_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_BASE_DIR=tmp_dir):
                self.assertFalse(self.backend.is_model_available(model=ModelName.MEDIUM))

    def test_ensure_model_available_skips_download_when_already_available(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_BASE_DIR=tmp_dir):
                model_dir = self.backend.get_local_model_dir(model=ModelName.MEDIUM)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.json").write_text("{}")
                (model_dir / "weights.npz").write_bytes(b"fake")

                with patch("ml.backends.mlx_whisper_backend.snapshot_download", create=True) as mock_download:
                    self.backend.ensure_model_available(model=ModelName.MEDIUM)
                    mock_download.assert_not_called()

    def test_ensure_model_available_downloads_when_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_BASE_DIR=tmp_dir):
                def fake_snapshot_download(**kwargs):
                    local_dir = Path(kwargs["local_dir"])
                    local_dir.mkdir(parents=True, exist_ok=True)
                    (local_dir / "config.json").write_text("{}")
                    (local_dir / "weights.npz").write_bytes(b"fake")

                with patch(
                        "huggingface_hub.snapshot_download",
                        side_effect=fake_snapshot_download,
                ) as mock_download:
                    self.backend.ensure_model_available(model=ModelName.MEDIUM)

                self.assertTrue(self.backend.is_model_available(model=ModelName.MEDIUM))
                mock_download.assert_called_once()

    def test_ensure_model_available_raises_clear_error_when_download_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_BASE_DIR=tmp_dir):
                with patch(
                        "huggingface_hub.snapshot_download",
                        side_effect=RuntimeError("download boom"),
                ):
                    with self.assertRaises(ModelDownloadError):
                        self.backend.ensure_model_available(model=ModelName.MEDIUM)

    def test_load_model_returns_handle(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with override_settings(MODELS_BASE_DIR=tmp_dir):
                model_dir = self.backend.get_local_model_dir(model=ModelName.MEDIUM)
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.json").write_text("{}")
                (model_dir / "weights.npz").write_bytes(b"fake")

                handle = self.backend.load_model(model=ModelName.MEDIUM)

        self.assertIsInstance(handle, MlxWhisperLoadedModelHandle)
        self.assertEqual(handle.backend_name, BackendName.MLX_WHISPER)
        self.assertEqual(handle.model_name, ModelName.MEDIUM)

    def test_transcribe_rejects_missing_audio_file(self) -> None:
        handle = MlxWhisperLoadedModelHandle(
            _model_name=ModelName.MEDIUM,
            local_model_dir=Path("/tmp/fake-model"),
        )

        with self.assertRaises(FileNotFoundError):
            self.backend.transcribe(
                loaded_model=handle,
                audio_path=Path("/tmp/does-not-exist.wav"),
            )

    def test_transcribe_returns_transcription_result(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "chunk.wav"
            audio_path.write_bytes(b"fake-audio")

            handle = MlxWhisperLoadedModelHandle(
                _model_name=ModelName.MEDIUM,
                local_model_dir=Path(tmp_dir) / "mlx-model",
            )

            fake_result = {
                "text": "hello world",
                "segments": [
                    {
                        "words": [
                            {
                                "word": "hello",
                                "start": 0.1,
                                "end": 0.5,
                                "probability": 0.9,
                            },
                            {
                                "word": "world",
                                "start": 0.6,
                                "end": 1.0,
                                "probability": 0.8,
                            },
                        ]
                    }
                ],
            }

            fake_module = Mock()
            fake_module.transcribe.return_value = fake_result

            with patch.dict("sys.modules", {"mlx_whisper": fake_module}):
                result = self.backend.transcribe(
                    loaded_model=handle,
                    audio_path=audio_path,
                )

        self.assertEqual(result.full_text, "hello world")
        self.assertEqual(result.model_used, ModelName.MEDIUM)
        self.assertEqual(result.backend_used, BackendName.MLX_WHISPER)
        self.assertEqual(len(result.words), 2)
        self.assertEqual(result.words[0].text, "hello")
        self.assertEqual(result.words[0].start_time, 100)
        self.assertEqual(result.words[0].end_time, 500)

    def test_transcribe_raises_clear_error_when_backend_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "chunk.wav"
            audio_path.write_bytes(b"fake-audio")

            handle = MlxWhisperLoadedModelHandle(
                _model_name=ModelName.MEDIUM,
                local_model_dir=Path(tmp_dir) / "mlx-model",
            )

            fake_module = Mock()
            fake_module.transcribe.side_effect = RuntimeError("mlx boom")

            with patch.dict("sys.modules", {"mlx_whisper": fake_module}):
                with self.assertRaises(TranscriptionBackendError):
                    self.backend.transcribe(
                        loaded_model=handle,
                        audio_path=audio_path,
                    )


class ModelManagerTests(SimpleTestCase):
    def setUp(self) -> None:
        self.manager = ModelManager()

    def test_get_default_model_returns_medium(self) -> None:
        self.assertEqual(self.manager.get_default_model(), ModelName.MEDIUM)

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_get_default_backend_returns_mlx_on_apple_silicon(self, mock_machine, mock_system) -> None:
        self.assertEqual(self.manager.get_default_backend(), BackendName.MLX_WHISPER)

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_get_default_backend_returns_whisper_cpp_on_linux(self, mock_machine, mock_system) -> None:
        self.assertEqual(self.manager.get_default_backend(), BackendName.WHISPER_CPP)

    def test_resolve_selection_uses_defaults(self) -> None:
        selection = self.manager.resolve_selection()
        self.assertEqual(selection.model, ModelName.MEDIUM)

    def test_resolve_selection_rejects_unsupported_backend(self) -> None:
        with self.assertRaises(UnsupportedBackendError):
            self.manager.resolve_selection(backend="bad-backend")  # type: ignore[arg-type]

    def test_get_backend_returns_mlx_backend(self) -> None:
        backend = self.manager.get_backend(backend=BackendName.MLX_WHISPER)
        self.assertEqual(backend.name, BackendName.MLX_WHISPER)

    def test_get_backend_raises_for_unimplemented_whisper_cpp(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.manager.get_backend(backend=BackendName.WHISPER_CPP)

    def test_is_model_available_delegates_to_backend(self) -> None:
        backend = Mock()
        backend.is_model_available.return_value = True

        with patch.object(self.manager, "get_backend", return_value=backend):
            available = self.manager.is_model_available(
                backend=BackendName.MLX_WHISPER,
                model=ModelName.MEDIUM,
            )

        self.assertTrue(available)
        backend.is_model_available.assert_called_once_with(model=ModelName.MEDIUM)

    def test_ensure_model_available_delegates_to_backend(self) -> None:
        backend = Mock()

        with patch.object(self.manager, "get_backend", return_value=backend):
            selection = self.manager.ensure_model_available(
                backend=BackendName.MLX_WHISPER,
                model=ModelName.MEDIUM,
            )

        self.assertEqual(selection.backend, BackendName.MLX_WHISPER)
        self.assertEqual(selection.model, ModelName.MEDIUM)
        backend.ensure_model_available.assert_called_once_with(model=ModelName.MEDIUM)

    def test_load_model_returns_selection_backend_and_handle(self) -> None:
        backend = Mock()
        handle = Mock()

        with patch.object(
                self.manager,
                "ensure_model_available",
                return_value=ResolvedModelSelection(
                    backend=BackendName.MLX_WHISPER,
                    model=ModelName.MEDIUM,
                ),
        ):
            with patch.object(self.manager, "get_backend", return_value=backend):
                backend.load_model.return_value = handle

                selection, backend_impl, loaded_handle = self.manager.load_model(
                    backend=BackendName.MLX_WHISPER,
                    model=ModelName.MEDIUM,
                )

        self.assertEqual(selection.backend, BackendName.MLX_WHISPER)
        self.assertEqual(selection.model, ModelName.MEDIUM)
        self.assertIs(backend_impl, backend)
        self.assertIs(loaded_handle, handle)
        backend.load_model.assert_called_once_with(model=ModelName.MEDIUM)