from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from pipelines.services import upload_and_normalize_recording, upload_normalize_and_diarize_recording, \
    upload_normalize_diarize_and_vad_recording
from recordings.models import RecordingStatus, Recording


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