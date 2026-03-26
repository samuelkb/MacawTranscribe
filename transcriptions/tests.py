import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock, MagicMock

from django.test import TestCase, SimpleTestCase
from django.urls import reverse

from ml.backends.base import TranscribedWord, TranscriptionResult, TranscriptionBackend, LoadedModelHandle
from ml.types import BackendName, ModelName
from recordings.models import Recording, RecordingStatus, Chunk, ChunkStatus
from transcriptions.models import TranscriptWord, Transcript, TranscriptCandidate
from transcriptions.runtime import LoadedWorkerRuntime
from transcriptions.services import persist_transcription_words, create_initial_transcript, create_transcript_candidate, \
    apply_candidate, append_edit, transcribe_chunk, ChunkTranscriptionError, load_worker_transcription_runtime


class TranscriptionServicesTest(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

        self.recording_dir = Path(self.tmp_dir.name) / "recording"
        self.recording_dir.mkdir(parents=True, exist_ok=True)

        self.normalized_path = self.recording_dir / "normalized.wav"
        self.normalized_path.write_bytes(b"fake-normalized-audio")

        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path=str(self.recording_dir / "original.m4a"),
            normalized_file_path=str(self.normalized_path),
            duration_milliseconds=60_000,
            status=RecordingStatus.CHUNKED,
        )

        self.chunk = Chunk.objects.create(
            recording=self.recording,
            chunk_index=0,
            start_time=0,
            end_time=30_000,
            status=ChunkStatus.PENDING,
        )

    def test_persist_transcription_words_replaces_existing_words(self) -> None:
        TranscriptWord.objects.create(
            chunk=self.chunk,
            word_index=0,
            text="old",
            start_time=0,
            end_time=100,
            confidence=0.5,
            model_used="medium",
        )

        words = (
            TranscribedWord(
                word_index=0,
                text="hello",
                start_time=100,
                end_time=300,
                confidence=0.9,
            ),
            TranscribedWord(
                word_index=1,
                text="world",
                start_time=350,
                end_time=700,
                confidence=0.8,
            ),
        )

        created = persist_transcription_words(
            chunk=self.chunk,
            words=words,
            model_used="medium",
        )

        self.assertEqual(len(created), 2)

        persisted = list(
            TranscriptWord.objects.filter(chunk=self.chunk)
            .order_by("word_index")
            .values_list("word_index", "text", "start_time", "end_time", "confidence", "model_used")
        )
        self.assertEqual(
            persisted,
            [
                (0, "hello", 100, 300, 0.9, "medium"),
                (1, "world", 350, 700, 0.8, "medium"),
            ],
        )

    def test_create_initial_transcript_creates_transcript(self) -> None:
        transcript = create_initial_transcript(
            chunk=self.chunk,
            accepted_text="hello world",
            model_used="medium",
        )

        self.assertEqual(transcript.chunk, self.chunk)
        self.assertEqual(transcript.accepted_text, "hello world")
        self.assertEqual(transcript.model_used, "medium")

    def test_create_initial_transcript_rejects_existing_transcript(self) -> None:
        Transcript.objects.create(
            chunk=self.chunk,
            accepted_text="existing",
            model_used="medium",
        )

        with self.assertRaisesMessage(ValueError, "chunk already has an accepted transcript"):
            create_initial_transcript(
                chunk=self.chunk,
                accepted_text="new",
                model_used="large-v3",
            )

    def test_create_transcript_candidate_creates_candidate_and_marks_chunk_pending(self) -> None:
        candidate = create_transcript_candidate(
            chunk=self.chunk,
            candidate_text="candidate text",
            model_used="large-v3",
            confidence=0.77,
            is_from_retry=True,
        )

        self.assertEqual(candidate.chunk, self.chunk)
        self.assertEqual(candidate.candidate_text, "candidate text")
        self.assertEqual(candidate.model_used, "large-v3")
        self.assertEqual(candidate.confidence, 0.77)
        self.assertTrue(candidate.is_from_retry)

        self.chunk.refresh_from_db()
        self.assertTrue(self.chunk.has_pending_candidate)

    def test_apply_candidate_updates_transcript_and_rejects_other_candidates(self) -> None:
        transcript = Transcript.objects.create(
            chunk=self.chunk,
            accepted_text="old accepted",
            model_used="medium",
        )

        candidate_1 = TranscriptCandidate.objects.create(
            chunk=self.chunk,
            candidate_text="candidate one",
            model_used="large-v3",
        )
        candidate_2 = TranscriptCandidate.objects.create(
            chunk=self.chunk,
            candidate_text="candidate two",
            model_used="medium",
        )

        self.chunk.has_pending_candidate = True
        self.chunk.save(update_fields=["has_pending_candidate"])

        updated_transcript = apply_candidate(candidate=candidate_2)

        updated_transcript.refresh_from_db()
        candidate_1.refresh_from_db()
        candidate_2.refresh_from_db()
        self.chunk.refresh_from_db()

        self.assertEqual(updated_transcript.accepted_text, "candidate two")
        self.assertEqual(updated_transcript.model_used, "medium")

        self.assertTrue(candidate_2.accepted)
        self.assertFalse(candidate_2.rejected)
        self.assertIsNotNone(candidate_2.accepted_at)

        self.assertFalse(candidate_1.accepted)
        self.assertTrue(candidate_1.rejected)
        self.assertIsNotNone(candidate_1.rejected_at)

        self.assertFalse(self.chunk.has_pending_candidate)

    def test_append_edit_appends_history_and_updates_transcript(self) -> None:
        transcript = Transcript.objects.create(
            chunk=self.chunk,
            accepted_text="old accepted",
            model_used="medium",
        )

        edit = append_edit(
            transcript=transcript,
            edited_text="human edited text",
            editor="user",
        )

        transcript.refresh_from_db()

        self.assertEqual(edit.transcript, transcript)
        self.assertEqual(edit.edited_text, "human edited text")
        self.assertEqual(edit.editor, "user")
        self.assertEqual(transcript.accepted_text, "human edited text")

    @patch("transcriptions.services.extract_chunk_audio")
    @patch("transcriptions.services.ModelManager")
    def test_transcribe_chunk_happy_path_creates_words_and_initial_transcript(
            self,
            mock_model_manager_cls: Mock,
            mock_extract_chunk_audio: Mock,
    ) -> None:
        temp_chunk_audio = Path(self.tmp_dir.name) / "chunk.wav"
        temp_chunk_audio.write_bytes(b"fake-chunk-audio")
        mock_extract_chunk_audio.return_value = temp_chunk_audio

        mock_backend = Mock()
        mock_loaded_model = Mock()
        mock_manager = Mock()
        mock_model_manager_cls.return_value = mock_manager
        mock_manager.load_model.return_value = (
            Mock(backend=BackendName.MLX_WHISPER, model=ModelName.MEDIUM),
            mock_backend,
            mock_loaded_model,
        )

        mock_backend.transcribe.return_value = TranscriptionResult(
            full_text="hello world",
            words=(
                TranscribedWord(
                    word_index=0,
                    text="hello",
                    start_time=0,
                    end_time=250,
                    confidence=0.9,
                ),
                TranscribedWord(
                    word_index=1,
                    text="world",
                    start_time=300,
                    end_time=650,
                    confidence=0.8,
                ),
            ),
            model_used=ModelName.MEDIUM,
            backend_used=BackendName.MLX_WHISPER,
        )

        transcript = transcribe_chunk(
            chunk_id=self.chunk.id,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            worker_id="worker-1",
        )

        self.chunk.refresh_from_db()

        self.assertEqual(self.chunk.status, ChunkStatus.COMPLETED)
        self.assertEqual(self.chunk.attempt_count, 1)
        self.assertEqual(self.chunk.worker_id, "worker-1")
        self.assertEqual(transcript.accepted_text, "hello world")
        self.assertEqual(transcript.model_used, "medium")

        words = list(
            TranscriptWord.objects.filter(chunk=self.chunk)
            .order_by("word_index")
            .values_list("word_index", "text", "model_used")
        )
        self.assertEqual(
            words,
            [
                (0, "hello", "medium"),
                (1, "world", "medium"),
            ],
        )

        self.assertFalse(temp_chunk_audio.exists())

    @patch("transcriptions.services.extract_chunk_audio")
    @patch("transcriptions.services.ModelManager")
    def test_transcribe_chunk_creates_candidate_when_transcript_already_exists(
            self,
            mock_model_manager_cls: Mock,
            mock_extract_chunk_audio: Mock,
    ) -> None:
        Transcript.objects.create(
            chunk=self.chunk,
            accepted_text="existing accepted text",
            model_used="medium",
        )

        temp_chunk_audio = Path(self.tmp_dir.name) / "chunk.wav"
        temp_chunk_audio.write_bytes(b"fake-chunk-audio")
        mock_extract_chunk_audio.return_value = temp_chunk_audio

        mock_backend = Mock()
        mock_loaded_model = Mock()
        mock_manager = Mock()
        mock_model_manager_cls.return_value = mock_manager
        mock_manager.load_model.return_value = (
            Mock(backend=BackendName.MLX_WHISPER, model=ModelName.LARGE_V3),
            mock_backend,
            mock_loaded_model,
        )

        mock_backend.transcribe.return_value = TranscriptionResult(
            full_text="new candidate text",
            words=(
                TranscribedWord(
                    word_index=0,
                    text="new",
                    start_time=0,
                    end_time=100,
                    confidence=0.9,
                ),
            ),
            model_used=ModelName.LARGE_V3,
            backend_used=BackendName.MLX_WHISPER,
        )

        transcript = transcribe_chunk(
            chunk_id=self.chunk.id,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.LARGE_V3,
            worker_id="worker-1",
        )

        self.chunk.refresh_from_db()
        transcript.refresh_from_db()

        self.assertEqual(transcript.accepted_text, "existing accepted text")
        self.assertEqual(TranscriptCandidate.objects.filter(chunk=self.chunk).count(), 1)

        candidate = TranscriptCandidate.objects.get(chunk=self.chunk)
        self.assertEqual(candidate.candidate_text, "new candidate text")
        self.assertEqual(candidate.model_used, "large-v3")
        self.assertTrue(self.chunk.has_pending_candidate)
        self.assertEqual(self.chunk.status, ChunkStatus.COMPLETED)

        self.assertFalse(temp_chunk_audio.exists())

    @patch("transcriptions.services.extract_chunk_audio")
    @patch("transcriptions.services.ModelManager")
    def test_transcribe_chunk_marks_chunk_failed_on_backend_error(
            self,
            mock_model_manager_cls: Mock,
            mock_extract_chunk_audio: Mock,
    ) -> None:
        temp_chunk_audio = Path(self.tmp_dir.name) / "chunk.wav"
        temp_chunk_audio.write_bytes(b"fake-chunk-audio")
        mock_extract_chunk_audio.return_value = temp_chunk_audio

        mock_backend = Mock()
        mock_loaded_model = Mock()
        mock_manager = Mock()
        mock_model_manager_cls.return_value = mock_manager
        mock_manager.load_model.return_value = (
            Mock(backend=BackendName.MLX_WHISPER, model=ModelName.MEDIUM),
            mock_backend,
            mock_loaded_model,
        )

        mock_backend.transcribe.side_effect = RuntimeError("backend boom")

        with self.assertRaisesMessage(ChunkTranscriptionError, "backend boom"):
            transcribe_chunk(
                chunk_id=self.chunk.id,
                backend=BackendName.MLX_WHISPER,
                model=ModelName.MEDIUM,
                worker_id="worker-1",
            )

        self.chunk.refresh_from_db()

        self.assertEqual(self.chunk.status, ChunkStatus.FAILED)
        self.assertEqual(self.chunk.attempt_count, 1)
        self.assertEqual(self.chunk.last_error, "backend boom")
        self.assertIsNotNone(self.chunk.last_failed_at)
        self.assertFalse(temp_chunk_audio.exists())


    @patch("transcriptions.services.extract_chunk_audio")
    @patch("transcriptions.services.ModelManager")
    def test_transcribe_chunk_logs_start_and_completion(
            self,
            mock_model_manager_cls: Mock,
            mock_extract_chunk_audio: Mock,
    ) -> None:
        temp_chunk_audio = Path(self.tmp_dir.name) / "chunk.wav"
        temp_chunk_audio.write_bytes(b"fake-chunk-audio")
        mock_extract_chunk_audio.return_value = temp_chunk_audio

        mock_backend = Mock()
        mock_loaded_model = Mock()
        mock_manager = Mock()
        mock_model_manager_cls.return_value = mock_manager
        mock_manager.load_model.return_value = (
            Mock(backend=BackendName.MLX_WHISPER, model=ModelName.MEDIUM),
            mock_backend,
            mock_loaded_model,
        )

        mock_backend.transcribe.return_value = TranscriptionResult(
            full_text="hello world",
            words=(
                TranscribedWord(
                    word_index=0,
                    text="hello",
                    start_time=0,
                    end_time=100,
                    confidence=0.9,
                ),
            ),
            model_used=ModelName.MEDIUM,
            backend_used=BackendName.MLX_WHISPER,
        )

        with self.assertLogs("transcriptions.services", level="INFO") as captured:
            transcribe_chunk(
                chunk_id=self.chunk.id,
                backend=BackendName.MLX_WHISPER,
                model=ModelName.MEDIUM,
                worker_id="worker-1",
            )

        output = "\n".join(captured.output)
        self.assertIn("chunk_transcription_started", output)
        self.assertIn("chunk_transcription_completed", output)

        temp_chunk_audio.unlink(missing_ok=True)


class TranscribeChunkViewTests(TestCase):
    def setUp(self) -> None:
        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=60_000,
            status=RecordingStatus.CHUNKED,
        )

        self.chunk = Chunk.objects.create(
            recording=self.recording,
            chunk_index=0,
            start_time=0,
            end_time=30_000,
            status=ChunkStatus.PENDING,
        )

    def test_returns_201_and_payload_on_success(self) -> None:
        transcript = Transcript.objects.create(
            chunk=self.chunk,
            accepted_text="hello world",
            model_used="medium",
        )

        self.chunk.status = ChunkStatus.COMPLETED
        self.chunk.save(update_fields=["status"])

        with patch(
                "transcriptions.views.transcribe_chunk",
                return_value=transcript,
        ):
            response = self.client.post(
                reverse("transcriptions:transcribe_chunk", args=[self.chunk.id]),
                data=json.dumps(
                    {
                        "backend": "mlx-whisper",
                        "model": "medium",
                        "worker_id": "worker-1",
                    }
                ),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()

        self.assertEqual(payload["chunk_id"], str(self.chunk.id))
        self.assertEqual(payload["recording_id"], str(self.recording.id))
        self.assertEqual(payload["status"], ChunkStatus.COMPLETED)
        self.assertEqual(payload["accepted_text"], "hello world")
        self.assertEqual(payload["model_used"], "medium")
        self.assertFalse(payload["has_pending_candidate"])
        self.assertEqual(payload["word_count"], 0)

    def test_returns_400_on_invalid_json(self) -> None:
        response = self.client.post(
            reverse("transcriptions:transcribe_chunk", args=[self.chunk.id]),
            data="{bad json",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_json")

    def test_returns_400_on_invalid_backend(self) -> None:
        response = self.client.post(
            reverse("transcriptions:transcribe_chunk", args=[self.chunk.id]),
            data=json.dumps({"backend": "bad-backend"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_backend_or_model")

    def test_returns_400_when_service_rejects_request(self) -> None:
        with patch(
                "transcriptions.views.transcribe_chunk",
                side_effect=ChunkTranscriptionError("backend boom"),
        ):
            response = self.client.post(
                reverse("transcriptions:transcribe_chunk", args=[self.chunk.id]),
                data=json.dumps({"backend": "mlx-whisper", "model": "medium"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "chunk_transcription_failed")
        self.assertEqual(response.json()["detail"], "backend boom")

    def test_returns_500_on_unexpected_failure(self) -> None:
        with patch(
                "transcriptions.views.transcribe_chunk",
                side_effect=RuntimeError("boom"),
        ):
            response = self.client.post(
                reverse("transcriptions:transcribe_chunk", args=[self.chunk.id]),
                data=json.dumps({"backend": "mlx-whisper", "model": "medium"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "unexpected_error")

    def test_logs_unexpected_failure(self) -> None:
        with patch(
                "transcriptions.views.transcribe_chunk",
                side_effect=RuntimeError("boom"),
        ):
            with self.assertLogs("transcriptions.views", level="ERROR") as captured:
                self.client.post(
                    reverse("transcriptions:transcribe_chunk", args=[self.chunk.id]),
                    data=json.dumps({"backend": "mlx-whisper", "model": "medium"}),
                    content_type="application/json",
                )

        output = "\n".join(captured.output)
        self.assertIn("chunk_transcription_endpoint_failed", output)


class LoadedWorkerRuntimeTests(SimpleTestCase):
    def test_partition_key(self) -> None:
        backend_impl = self._make_backend_stub()
        loaded_model = self._make_model_stub()

        runtime = LoadedWorkerRuntime(
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            backend_impl=backend_impl,
            loaded_model=loaded_model,
        )

        self.assertEqual(runtime.partition_key, "mlx-whisper:medium")

    def _make_backend_stub(self) -> TranscriptionBackend:
        class BackendStub(TranscriptionBackend):
            @property
            def name(self) -> BackendName:
                return BackendName.MLX_WHISPER

            def supports_model(self, *, model: ModelName) -> bool:
                return True

            def is_model_available(self, *, model: ModelName) -> bool:
                return True

            def ensure_model_available(self, *, model: ModelName) -> None:
                return None

            def load_model(self, *, model: ModelName) -> LoadedModelHandle:
                raise NotImplementedError

            def transcribe(self, *, loaded_model: LoadedModelHandle, audio_path):
                raise NotImplementedError

        return BackendStub()

    def _make_model_stub(self) -> LoadedModelHandle:
        class ModelStub(LoadedModelHandle):
            @property
            def backend_name(self) -> BackendName:
                return BackendName.MLX_WHISPER

            @property
            def model_name(self) -> ModelName:
                return ModelName.MEDIUM

        return ModelStub()


class LoadWorkerTranscriptionRuntimeTests(SimpleTestCase):
    @patch("transcriptions.services.ModelManager")
    def test_load_worker_transcription_runtime_returns_loaded_runtime(self, model_manager_cls) -> None:
        manager = model_manager_cls.return_value

        selection = MagicMock()
        selection.backend = BackendName.MLX_WHISPER
        selection.model = ModelName.MEDIUM

        backend_impl = MagicMock()
        loaded_model = MagicMock()

        manager.load_model.return_value = (selection, backend_impl, loaded_model)

        runtime = load_worker_transcription_runtime(
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        manager.load_model.assert_called_once_with(
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        self.assertIsInstance(runtime, LoadedWorkerRuntime)
        self.assertEqual(runtime.backend, BackendName.MLX_WHISPER)
        self.assertEqual(runtime.model, ModelName.MEDIUM)
        self.assertIs(runtime.backend_impl, backend_impl)
        self.assertIs(runtime.loaded_model, loaded_model)
        self.assertEqual(runtime.partition_key, "mlx-whisper:medium")

    @patch("transcriptions.services.ModelManager")
    def test_load_worker_transcription_runtime_uses_resolved_selection(self, model_manager_cls) -> None:
        manager = model_manager_cls.return_value

        selection = MagicMock()
        selection.backend = BackendName.MLX_WHISPER
        selection.model = ModelName.LARGE_V3

        backend_impl = MagicMock()
        loaded_model = MagicMock()

        manager.load_model.return_value = (selection, backend_impl, loaded_model)

        runtime = load_worker_transcription_runtime(
            backend=None,
            model=None,
        )

        manager.load_model.assert_called_once_with(
            backend=None,
            model=None,
        )

        self.assertEqual(runtime.backend, BackendName.MLX_WHISPER)
        self.assertEqual(runtime.model, ModelName.LARGE_V3)
