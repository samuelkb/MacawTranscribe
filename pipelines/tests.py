import json
import threading
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4, UUID

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings, SimpleTestCase
from django.urls import reverse
from django.utils import timezone

from ml.types import BackendName, ModelName
from pipelines.queue import enqueue_transcription_job, dequeue_transcription_job, QueueError
from pipelines.queue_types import TranscriptionJob
from pipelines.services import upload_and_normalize_recording, upload_normalize_and_diarize_recording, \
    upload_normalize_diarize_and_vad_recording, upload_normalize_diarize_vad_and_chunk_recording, queue_jobs
from pipelines.worker import generate_worker_id, update_chunk_heartbeat, recover_stale_processing_chunks, \
    run_worker_loop, WorkerConfig, process_transcription_job, _should_recycle_worker_after_job
from recordings.models import RecordingStatus, Recording, Chunk, ChunkStatus
from transcriptions.runtime import LoadedWorkerRuntime
from user_settings.models import WorkerProcessState, WorkerRole, WorkerStatus, TranscriptionRuntimeSettings
from user_settings.services import register_worker_process


class TranscriptionJobTests(SimpleTestCase):
    def test_roundtrip_serialization(self) -> None:
        job = TranscriptionJob(
            chunk_id=UUID("11111111-1111-1111-1111-111111111111"),
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        payload = job.to_dict()
        restored = TranscriptionJob.from_dict(payload)

        self.assertEqual(str(restored.chunk_id), "11111111-1111-1111-1111-111111111111")
        self.assertEqual(restored.backend, BackendName.MLX_WHISPER)
        self.assertEqual(restored.model, ModelName.MEDIUM)


class QueueHelpersTests(SimpleTestCase):
    @patch("pipelines.queue.get_redis_client")
    def test_enqueue_transcription_job_pushes_json(self, mock_get_redis_client: Mock) -> None:
        mock_client = Mock()
        mock_get_redis_client.return_value = mock_client

        job = TranscriptionJob(
            chunk_id=UUID("11111111-1111-1111-1111-111111111111"),
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        enqueue_transcription_job(job=job)

        mock_client.rpush.assert_called_once()
        queue_name, payload = mock_client.rpush.call_args.args
        self.assertEqual(queue_name, "transcription_jobs")
        self.assertEqual(
            json.loads(payload),
            {
                "chunk_id": "11111111-1111-1111-1111-111111111111",
                "backend": "mlx-whisper",
                "model": "medium",
            },
        )

    @patch("pipelines.queue.get_redis_client")
    def test_dequeue_transcription_job_returns_none_on_timeout(self, mock_get_redis_client: Mock) -> None:
        mock_client = Mock()
        mock_client.blpop.return_value = None
        mock_get_redis_client.return_value = mock_client

        job = dequeue_transcription_job(timeout_seconds=1)

        self.assertIsNone(job)

    @patch("pipelines.queue.get_redis_client")
    def test_dequeue_transcription_job_parses_payload(self, mock_get_redis_client: Mock) -> None:
        mock_client = Mock()
        mock_client.blpop.return_value = (
            "transcription_jobs",
            json.dumps(
                {
                    "chunk_id": "11111111-1111-1111-1111-111111111111",
                    "backend": "mlx-whisper",
                    "model": "medium",
                }
            ),
        )
        mock_get_redis_client.return_value = mock_client

        job = dequeue_transcription_job(timeout_seconds=1)

        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(str(job.chunk_id), "11111111-1111-1111-1111-111111111111")
        self.assertEqual(job.backend, BackendName.MLX_WHISPER)
        self.assertEqual(job.model, ModelName.MEDIUM)

    @patch("pipelines.queue.get_redis_client")
    def test_dequeue_transcription_job_raises_on_invalid_payload(self, mock_get_redis_client: Mock) -> None:
        mock_client = Mock()
        mock_client.blpop.return_value = ("transcription_jobs", "{bad json")
        mock_get_redis_client.return_value = mock_client

        with self.assertRaises(QueueError):
            dequeue_transcription_job(timeout_seconds=1)


class QueueJobsTests(TestCase):
    def setUp(self) -> None:
        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=60_000,
            status=RecordingStatus.CHUNKED,
        )

        self.chunk_0 = Chunk.objects.create(
            recording=self.recording,
            chunk_index=0,
            start_time=0,
            end_time=30_000,
            status=ChunkStatus.PENDING,
        )
        self.chunk_1 = Chunk.objects.create(
            recording=self.recording,
            chunk_index=1,
            start_time=25_000,
            end_time=55_000,
            status=ChunkStatus.PENDING,
        )

    @patch("pipelines.services.enqueue_transcription_job")
    def test_queue_jobs_enqueues_pending_chunks_and_marks_recording_transcribing(self, mock_enqueue) -> None:
        count = queue_jobs(
            recording=self.recording,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        self.assertEqual(count, 2)

        self.chunk_0.refresh_from_db()
        self.chunk_1.refresh_from_db()
        self.recording.refresh_from_db()

        self.assertEqual(self.chunk_0.status, ChunkStatus.QUEUED)
        self.assertEqual(self.chunk_1.status, ChunkStatus.QUEUED)
        self.assertEqual(self.recording.status, RecordingStatus.TRANSCRIBING)

        self.assertEqual(mock_enqueue.call_count, 2)

        first_job = mock_enqueue.call_args_list[0].kwargs["job"]
        self.assertEqual(first_job.chunk_id, self.chunk_0.id)
        self.assertEqual(first_job.backend, BackendName.MLX_WHISPER)
        self.assertEqual(first_job.model, ModelName.MEDIUM)

    def test_queue_jobs_rejects_non_chunked_recording(self) -> None:
        self.recording.status = RecordingStatus.DIARIZED
        self.recording.save(update_fields=["status"])

        with self.assertRaisesMessage(
                ValueError,
                "recording must be in chunked status before queueing jobs",
        ):
            queue_jobs(recording=self.recording)

    def test_queue_jobs_rejects_when_no_eligible_chunks(self) -> None:
        self.chunk_0.status = ChunkStatus.COMPLETED
        self.chunk_0.save(update_fields=["status"])
        self.chunk_1.status = ChunkStatus.COMPLETED
        self.chunk_1.save(update_fields=["status"])

        with self.assertRaisesMessage(
                ValueError,
                "recording has no eligible chunks to queue",
        ):
            queue_jobs(recording=self.recording)

    @patch("pipelines.services.enqueue_transcription_job")
    def test_queue_jobs_can_include_failed_chunks(self, mock_enqueue) -> None:
        self.chunk_0.status = ChunkStatus.FAILED
        self.chunk_0.save(update_fields=["status"])
        self.chunk_1.status = ChunkStatus.PENDING
        self.chunk_1.save(update_fields=["status"])

        count = queue_jobs(
            recording=self.recording,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            include_failed=True,
        )

        self.assertEqual(count, 2)


class UploadAndNormalizeRecordingTests(TestCase):
    def test_upload_and_normalize_recording_returns_normalized_recording_on_success(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            uploaded = SimpleUploadedFile(
                "interview.m4a",
                b"fake audio bytes",
                content_type="audio/mp4",
            )

            with override_settings(RECORDINGS_BASE_DIR=tmp_dir):
                with patch(
                        "recordings.services.probe_audio_duration_milliseconds",
                        return_value=125_000,
                ):
                    with patch("recordings.services._run_ffmpeg_normalization"):
                        result = upload_and_normalize_recording(uploaded_file=uploaded)

            self.assertTrue(result.normalization_succeeded)
            self.assertIsNone(result.warning)
            self.assertEqual(result.recording.status, RecordingStatus.NORMALIZED)
            self.assertIsNotNone(result.recording.normalized_file_path)

            result.recording.refresh_from_db()
            self.assertEqual(result.recording.status, RecordingStatus.NORMALIZED)
            self.assertIsNotNone(result.recording.normalized_file_path)

    def test_upload_and_normalize_recording_returns_partial_success_when_normalization_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            uploaded = SimpleUploadedFile(
                "interview.m4a",
                b"fake audio bytes",
                content_type="audio/mp4",
            )

            with override_settings(RECORDINGS_BASE_DIR=tmp_dir):
                with patch(
                        "recordings.services.probe_audio_duration_milliseconds",
                        return_value=125_000,
                ):
                    with patch(
                            "recordings.services.normalize_audio",
                            side_effect=RuntimeError("normalization boom"),
                    ):
                        result = upload_and_normalize_recording(uploaded_file=uploaded)

            self.assertFalse(result.normalization_succeeded)
            self.assertEqual(
                result.warning,
                "Upload succeeded but normalization failed.",
            )

            self.assertEqual(Recording.objects.count(), 1)

            recording = result.recording
            recording.refresh_from_db()

            self.assertEqual(recording.status, RecordingStatus.UPLOADED)
            self.assertIsNone(recording.normalized_file_path)

            original_path = Path(recording.original_file_path)
            self.assertTrue(original_path.exists())

    def test_upload_and_normalize_recording_propagates_ingestion_failures(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.services.ingest_uploaded_recording",
                side_effect=RuntimeError("ingestion boom"),
        ):
            with self.assertRaisesMessage(RuntimeError, "ingestion boom"):
                upload_and_normalize_recording(uploaded_file=uploaded)


class UploadNormalizeAndDiarizeRecordingTests(TestCase):
    def test_upload_normalize_and_diarize_recording_returns_full_success(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            uploaded = SimpleUploadedFile(
                "interview.m4a",
                b"fake audio bytes",
                content_type="audio/mp4",
            )

            with override_settings(RECORDINGS_BASE_DIR=tmp_dir):
                with patch(
                        "recordings.services.probe_audio_duration_milliseconds",
                        return_value=125_000,
                ):
                    with patch("recordings.services._run_ffmpeg_normalization"):
                        with patch("speakers.services.run_pyannote_diarization") as mock_diarization:
                            mock_diarization.return_value = [
                                type(
                                    "Seg",
                                    (),
                                    {
                                        "speaker_id": "SPEAKER_00",
                                        "start_time": 0,
                                        "end_time": 5_000,
                                    },
                                )()
                            ]

                            result = upload_normalize_and_diarize_recording(
                                uploaded_file=uploaded,
                            )

            self.assertTrue(result.normalization_succeeded)
            self.assertTrue(result.diarization_succeeded)
            self.assertIsNone(result.warning)

            result.recording.refresh_from_db()
            self.assertEqual(result.recording.status, RecordingStatus.DIARIZED)
            self.assertIsNotNone(result.recording.normalized_file_path)

    def test_upload_normalize_and_diarize_recording_returns_partial_success_when_normalization_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            duration_milliseconds=125_000,
            status=RecordingStatus.UPLOADED,
        )

        with patch(
                "pipelines.services.ingest_uploaded_recording",
                return_value=recording,
        ):
            with patch(
                    "pipelines.services.normalize_audio",
                    side_effect=RuntimeError("normalization boom"),
            ):
                result = upload_normalize_and_diarize_recording(uploaded_file=uploaded)

        self.assertFalse(result.normalization_succeeded)
        self.assertFalse(result.diarization_succeeded)
        self.assertEqual(
            result.warning,
            "Upload succeeded but normalization failed.",
        )

    def test_upload_normalize_and_diarize_recording_returns_partial_success_when_diarization_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.NORMALIZED,
        )

        with patch(
                "pipelines.services.ingest_uploaded_recording",
                return_value=recording,
        ):
            with patch(
                    "pipelines.services.normalize_audio",
                    return_value=recording,
            ):
                with patch(
                        "pipelines.services.run_diarization",
                        side_effect=RuntimeError("diarization boom"),
                ):
                    result = upload_normalize_and_diarize_recording(uploaded_file=uploaded)

        self.assertTrue(result.normalization_succeeded)
        self.assertFalse(result.diarization_succeeded)
        self.assertEqual(
            result.warning,
            "Upload and normalization succeeded but diarization failed.",
        )

    def test_upload_normalize_and_diarize_recording_propagates_ingestion_failures(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.services.ingest_uploaded_recording",
                side_effect=RuntimeError("ingestion boom"),
        ):
            with self.assertRaisesMessage(RuntimeError, "ingestion boom"):
                upload_normalize_and_diarize_recording(uploaded_file=uploaded)


class UploadNormalizeAndDiarizeViewTests(TestCase):
    def test_pipeline_view_returns_201_and_payload_on_success(self) -> None:
        recording = Recording(
            id=uuid4(),
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        result = Mock()
        result.recording = recording
        result.normalization_succeeded = True
        result.diarization_succeeded = True
        result.warning = None

        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_and_diarize_recording",
                return_value=result,
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_and_diarize_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()

        self.assertEqual(payload["original_file_name"], "interview.m4a")
        self.assertEqual(payload["duration_milliseconds"], 125_000)
        self.assertEqual(payload["status"], RecordingStatus.DIARIZED)
        self.assertTrue(payload["normalization_succeeded"])
        self.assertTrue(payload["diarization_succeeded"])
        self.assertEqual(
            payload["normalized_file_path"],
            "data/recordings/test/normalized.wav",
        )

    def test_pipeline_view_returns_201_with_warning_on_partial_success(self) -> None:
        recording = Recording(
            id=uuid4(),
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.NORMALIZED,
        )

        result = Mock()
        result.recording = recording
        result.normalization_succeeded = True
        result.diarization_succeeded = False
        result.warning = "Upload and normalization succeeded but diarization failed."

        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_and_diarize_recording",
                return_value=result,
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_and_diarize_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()

        self.assertFalse(payload["diarization_succeeded"])
        self.assertEqual(
            payload["warning"],
            "Upload and normalization succeeded but diarization failed.",
        )

    def test_pipeline_view_returns_400_when_file_missing(self) -> None:
        response = self.client.post(
            reverse("pipelines:upload_normalize_and_diarize_recording"),
            {},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "missing_file")

    def test_pipeline_view_returns_500_on_unexpected_failure(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_and_diarize_recording",
                side_effect=RuntimeError("boom"),
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_and_diarize_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "uploaded_failed")


class UploadNormalizeDiarizeAndVadRecordingTests(TestCase):
    def test_returns_full_success(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", return_value=[]):
                    with patch("pipelines.services.run_vad", return_value=[]):
                        result = upload_normalize_diarize_and_vad_recording(
                            uploaded_file=uploaded
                        )

        self.assertTrue(result.normalization_succeeded)
        self.assertTrue(result.diarization_succeeded)
        self.assertTrue(result.vad_succeeded)
        self.assertIsNone(result.warning)

    def test_returns_partial_success_when_normalization_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            duration_milliseconds=125_000,
            status=RecordingStatus.UPLOADED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", side_effect=RuntimeError("boom")):
                result = upload_normalize_diarize_and_vad_recording(uploaded_file=uploaded)

        self.assertFalse(result.normalization_succeeded)
        self.assertFalse(result.diarization_succeeded)
        self.assertFalse(result.vad_succeeded)
        self.assertEqual(result.warning, "Upload succeeded but normalization failed.")

    def test_returns_partial_success_when_diarization_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.NORMALIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", side_effect=RuntimeError("boom")):
                    result = upload_normalize_diarize_and_vad_recording(uploaded_file=uploaded)

        self.assertTrue(result.normalization_succeeded)
        self.assertFalse(result.diarization_succeeded)
        self.assertFalse(result.vad_succeeded)
        self.assertEqual(
            result.warning,
            "Upload and normalization succeeded but diarization failed.",
        )

    def test_returns_partial_success_when_vad_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", return_value=[]):
                    with patch("pipelines.services.run_vad", side_effect=RuntimeError("boom")):
                        result = upload_normalize_diarize_and_vad_recording(uploaded_file=uploaded)

        self.assertTrue(result.normalization_succeeded)
        self.assertTrue(result.diarization_succeeded)
        self.assertFalse(result.vad_succeeded)
        self.assertEqual(
            result.warning,
            "Upload, normalization, and diarization succeeded but VAD failed.",
        )


class UploadNormalizeDiarizeAndVadViewTest(TestCase):
    def test_returns_201_and_payload_on_success(self) -> None:
        recording = Recording(
            id=uuid4(),
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        result = Mock()
        result.recording = recording
        result.normalization_succeeded = True
        result.diarization_succeeded = True
        result.vad_succeeded = True
        result.warning = None

        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_diarize_and_vad_recording",
                return_value=result,
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_diarize_and_vad_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()

        self.assertTrue(payload["normalization_succeeded"])
        self.assertTrue(payload["diarization_succeeded"])
        self.assertTrue(payload["vad_succeeded"])
        self.assertEqual(payload["status"], RecordingStatus.DIARIZED)

    def test_returns_201_with_warning_on_partial_success(self) -> None:
        recording = Recording(
            id=uuid4(),
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        result = Mock()
        result.recording = recording
        result.normalization_succeeded = True
        result.diarization_succeeded = True
        result.vad_succeeded = False
        result.warning = "Upload, normalization, and diarization succeeded but VAD failed."

        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_diarize_and_vad_recording",
                return_value=result,
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_diarize_and_vad_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertFalse(payload["vad_succeeded"])
        self.assertEqual(
            payload["warning"],
            "Upload, normalization, and diarization succeeded but VAD failed.",
        )


class UploadNormalizeDiarizeVadAndChunkRecordingTests(TestCase):
    def test_returns_full_success(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", return_value=[]):
                    with patch("pipelines.services.run_vad", return_value=[]):
                        with patch("pipelines.services.create_chunks", return_value=[]):
                            result = upload_normalize_diarize_vad_and_chunk_recording(
                                uploaded_file=uploaded,
                            )

        self.assertTrue(result.normalization_succeeded)
        self.assertTrue(result.diarization_succeeded)
        self.assertTrue(result.vad_succeeded)
        self.assertTrue(result.chunk_creation_succeeded)
        self.assertIsNone(result.warning)

    def test_returns_partial_success_when_normalization_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            duration_milliseconds=125_000,
            status=RecordingStatus.UPLOADED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", side_effect=RuntimeError("normalization boom")):
                result = upload_normalize_diarize_vad_and_chunk_recording(
                    uploaded_file=uploaded,
                )

        self.assertFalse(result.normalization_succeeded)
        self.assertFalse(result.diarization_succeeded)
        self.assertFalse(result.vad_succeeded)
        self.assertFalse(result.chunk_creation_succeeded)
        self.assertEqual(
            result.warning,
            "Upload succeeded but normalization failed.",
        )

    def test_returns_partial_success_when_diarization_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.NORMALIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", side_effect=RuntimeError("diarization boom")):
                    result = upload_normalize_diarize_vad_and_chunk_recording(
                        uploaded_file=uploaded,
                    )

        self.assertTrue(result.normalization_succeeded)
        self.assertFalse(result.diarization_succeeded)
        self.assertFalse(result.vad_succeeded)
        self.assertFalse(result.chunk_creation_succeeded)
        self.assertEqual(
            result.warning,
            "Upload and normalization succeeded but diarization failed.",
        )

    def test_returns_partial_success_when_vad_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", return_value=[]):
                    with patch("pipelines.services.run_vad", side_effect=RuntimeError("vad boom")):
                        result = upload_normalize_diarize_vad_and_chunk_recording(
                            uploaded_file=uploaded,
                        )

        self.assertTrue(result.normalization_succeeded)
        self.assertTrue(result.diarization_succeeded)
        self.assertFalse(result.vad_succeeded)
        self.assertFalse(result.chunk_creation_succeeded)
        self.assertEqual(
            result.warning,
            "Upload, normalization, and diarization succeeded but VAD failed.",
        )

    def test_returns_partial_success_when_chunk_creation_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        with patch("pipelines.services.ingest_uploaded_recording", return_value=recording):
            with patch("pipelines.services.normalize_audio", return_value=recording):
                with patch("pipelines.services.run_diarization", return_value=[]):
                    with patch("pipelines.services.run_vad", return_value=[]):
                        with patch("pipelines.services.create_chunks", side_effect=RuntimeError("chunk boom")):
                            result = upload_normalize_diarize_vad_and_chunk_recording(
                                uploaded_file=uploaded,
                            )

        self.assertTrue(result.normalization_succeeded)
        self.assertTrue(result.diarization_succeeded)
        self.assertTrue(result.vad_succeeded)
        self.assertFalse(result.chunk_creation_succeeded)
        self.assertEqual(
            result.warning,
            "Upload, normalization, diarization, and VAD succeeded but chunk creation failed.",
        )

    def test_propagates_ingestion_failures(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.services.ingest_uploaded_recording",
                side_effect=RuntimeError("ingestion boom"),
        ):
            with self.assertRaisesMessage(RuntimeError, "ingestion boom"):
                upload_normalize_diarize_vad_and_chunk_recording(uploaded_file=uploaded)


class UploadNormalizeDiarizeVadAndChunkRecordingViewTests(TestCase):
    def test_returns_201_and_payload_on_success(self) -> None:
        recording = Recording(
            id=uuid4(),
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.CHUNKED,
        )

        result = Mock()
        result.recording = recording
        result.normalization_succeeded = True
        result.diarization_succeeded = True
        result.vad_succeeded = True
        result.chunk_creation_succeeded = True
        result.warning = None

        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_diarize_vad_and_chunk_recording",
                return_value=result,
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_diarize_vad_and_chunk_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()

        self.assertEqual(payload["recording_id"], str(recording.id))
        self.assertEqual(payload["original_file_name"], "interview.m4a")
        self.assertEqual(payload["duration_milliseconds"], 125_000)
        self.assertEqual(payload["status"], RecordingStatus.CHUNKED)
        self.assertEqual(payload["original_file_path"], "data/recordings/test/original.m4a")
        self.assertEqual(payload["normalized_file_path"], "data/recordings/test/normalized.wav")
        self.assertTrue(payload["normalization_succeeded"])
        self.assertTrue(payload["diarization_succeeded"])
        self.assertTrue(payload["vad_succeeded"])
        self.assertTrue(payload["chunk_creation_succeeded"])
        self.assertNotIn("warning", payload)

    def test_returns_201_with_warning_on_partial_success(self) -> None:
        recording = Recording(
            id=uuid4(),
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=125_000,
            status=RecordingStatus.DIARIZED,
        )

        result = Mock()
        result.recording = recording
        result.normalization_succeeded = True
        result.diarization_succeeded = True
        result.vad_succeeded = True
        result.chunk_creation_succeeded = False
        result.warning = "Upload, normalization, diarization, and VAD succeeded but chunk creation failed."

        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_diarize_vad_and_chunk_recording",
                return_value=result,
        ):
            response = self.client.post(
                reverse("pipelines:upload_normalize_diarize_vad_and_chunk_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()

        self.assertEqual(payload["status"], RecordingStatus.DIARIZED)
        self.assertTrue(payload["normalization_succeeded"])
        self.assertTrue(payload["diarization_succeeded"])
        self.assertTrue(payload["vad_succeeded"])
        self.assertFalse(payload["chunk_creation_succeeded"])
        self.assertEqual(
            payload["warning"],
            "Upload, normalization, diarization, and VAD succeeded but chunk creation failed.",
        )

    def test_returns_400_when_file_missing(self) -> None:
        response = self.client.post(
            reverse("pipelines:upload_normalize_diarize_vad_and_chunk_recording"),
            {},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "missing_file")

    def test_logs_failure(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "pipelines.views.upload_normalize_diarize_vad_and_chunk_recording",
                side_effect=RuntimeError("boom"),
        ):
            with self.assertLogs("pipelines.views", level="ERROR") as captured:
                self.client.post(
                    reverse("pipelines:upload_normalize_diarize_vad_and_chunk_recording"),
                    {"file": uploaded},
                )

        output = "\n".join(captured.output)
        self.assertIn("pipeline_upload_failed", output)


class WorkerHelpersTests(TestCase):
    def setUp(self) -> None:
        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=60_000,
            status=RecordingStatus.TRANSCRIBING,
        )

    def test_generate_worker_id_returns_readable_value(self) -> None:
        worker_id = generate_worker_id()
        self.assertIn(":", worker_id)

    def test_update_chunk_heartbeat_updates_only_processing_chunk_with_same_worker(self) -> None:
        chunk = Chunk.objects.create(
            recording=self.recording,
            chunk_index=0,
            start_time=0,
            end_time=30_000,
            status=ChunkStatus.PROCESSING,
            worker_id="worker-1",
        )

        before = timezone.now()
        update_chunk_heartbeat(chunk_id=chunk.id, worker_id="worker-1")
        chunk.refresh_from_db()

        self.assertIsNotNone(chunk.heartbeat_at)
        assert chunk.heartbeat_at is not None
        self.assertGreaterEqual(chunk.heartbeat_at, before)

    def test_recover_stale_processing_chunks_marks_stale_chunks_failed(self) -> None:
        stale_chunk = Chunk.objects.create(
            recording=self.recording,
            chunk_index=0,
            start_time=0,
            end_time=30_000,
            status=ChunkStatus.PROCESSING,
            worker_id="worker-1",
            heartbeat_at=timezone.now() - timedelta(seconds=120),
        )

        fresh_chunk = Chunk.objects.create(
            recording=self.recording,
            chunk_index=1,
            start_time=30_000,
            end_time=60_000,
            status=ChunkStatus.PROCESSING,
            worker_id="worker-2",
            heartbeat_at=timezone.now(),
        )

        recovered_count = recover_stale_processing_chunks(stale_after_seconds=30)

        self.assertEqual(recovered_count, 1)

        stale_chunk.refresh_from_db()
        fresh_chunk.refresh_from_db()

        self.assertEqual(stale_chunk.status, ChunkStatus.FAILED)
        self.assertEqual(stale_chunk.last_error, "worker heartbeat expired")
        self.assertEqual(stale_chunk.worker_id, "")

        self.assertEqual(fresh_chunk.status, ChunkStatus.PROCESSING)

    @patch("pipelines.worker.increment_jobs_processed")
    @patch("pipelines.worker.transcribe_chunk_with_runtime")
    @patch("pipelines.worker.ChunkHeartbeatThread")
    def test_process_transcription_job_marks_worker_busy_then_idle(
            self,
            mock_heartbeat_thread: Mock,
            mock_transcribe: Mock,
            mock_increment_jobs_processed: Mock,
    ) -> None:
        worker = register_worker_process(
            worker_id="worker-1",
            pid=12345,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            hostname="host",
        )
        runtime = LoadedWorkerRuntime(
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            backend_impl=Mock(),
            loaded_model=Mock(),
        )
        job = TranscriptionJob(
            chunk_id=self.recording.id,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        def transcribe_side_effect(*args, **kwargs):
            worker_state = WorkerProcessState.objects.get(worker_id=worker.worker_id)
            self.assertEqual(worker_state.status, WorkerStatus.BUSY)
            self.assertEqual(worker_state.current_chunk_id, job.chunk_id)

        mock_transcribe.side_effect = transcribe_side_effect

        process_transcription_job(
            job=job,
            worker_id=worker.worker_id,
            config=WorkerConfig(heartbeat_interval_seconds=1),
            runtime=runtime,
        )

        worker.refresh_from_db()
        self.assertEqual(worker.status, WorkerStatus.IDLE)
        self.assertIsNone(worker.current_chunk_id)
        mock_increment_jobs_processed.assert_called_once_with(worker_id=worker.worker_id)

    def test_should_recycle_worker_after_job_returns_reason_at_threshold(self) -> None:
        settings = TranscriptionRuntimeSettings.get_solo()
        settings.max_job_per_worker = 5
        settings.save(update_fields=["max_job_per_worker", "updated_at"])
        worker = register_worker_process(
            worker_id="worker-threshold",
            pid=20010,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            hostname="host",
        )
        WorkerProcessState.objects.filter(worker_id=worker.worker_id).update(jobs_processed=5)

        recycle_reason = _should_recycle_worker_after_job(worker_id=worker.worker_id)

        self.assertEqual(recycle_reason, "max_jobs_per_worker_reached")


class WorkerLoopTests(TestCase):
    @patch("pipelines.worker.generate_worker_id", return_value="worker-1")
    @patch("pipelines.worker.process_transcription_job")
    @patch("pipelines.worker.dequeue_transcription_job")
    @patch("pipelines.worker.recover_stale_processing_chunks")
    def test_run_worker_loop_recovers_then_processes_one_job_and_stops(
            self,
            mock_recover: Mock,
            mock_dequeue: Mock,
            mock_process: Mock,
            mock_generate_worker_id: Mock,
    ) -> None:
        stop_event = threading.Event()

        job = TranscriptionJob(
            chunk_id=self._uuid("11111111-1111-1111-1111-111111111111"),
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
        )

        def dequeue_side_effect(*, timeout_seconds: int):
            if not stop_event.is_set():
                stop_event.set()
                return job
            return None

        mock_dequeue.side_effect = dequeue_side_effect

        run_worker_loop(
            config=WorkerConfig(
                heartbeat_interval_seconds=1,
                stale_after_seconds=30,
                dequeue_timeout_seconds=1,
                recover_stale_chunks_on_startup=True,
            ),
            stop_event=stop_event,
        )

        mock_recover.assert_called_once_with(stale_after_seconds=30)
        mock_process.assert_called_once()

        called_job = mock_process.call_args.kwargs["job"]
        self.assertEqual(called_job.chunk_id, job.chunk_id)
        self.assertEqual(mock_process.call_args.kwargs["worker_id"], "worker-1")

    @patch("pipelines.worker.generate_worker_id", return_value="worker-1")
    @patch("pipelines.worker.time.sleep")
    def test_run_worker_loop_handles_dequeue_failure_then_stops(
            self,
            mock_sleep: Mock,
            mock_generate_worker_id: Mock,
    ) -> None:
        from pipelines.queue import QueueError

        stop_event = threading.Event()

        with patch("pipelines.worker.dequeue_transcription_job") as mock_dequeue:
            def side_effect(*, timeout_seconds: int):
                stop_event.set()
                raise QueueError("queue boom")

            mock_dequeue.side_effect = side_effect

            run_worker_loop(
                config=WorkerConfig(recover_stale_chunks_on_startup=False),
                stop_event=stop_event,
            )

        mock_sleep.assert_called_once_with(1)

    @staticmethod
    def _uuid(value: str):
        from uuid import UUID
        return UUID(value)
