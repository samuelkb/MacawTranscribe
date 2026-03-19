from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from pipelines.services import upload_and_normalize_recording
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
