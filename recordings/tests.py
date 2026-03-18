import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.db import models
from django.test import TestCase, SimpleTestCase

from recordings.models import Chunk, Recording, RecordingStatus
from recordings.services import create_recording, normalize_audio
from recordings.audio import AudioNormalizationError, _run_ffmpeg_normalization


class ChunkMetaTests(TestCase):
    def test_constraints_are_constraint_objects(self) -> None:
        self.assertIsInstance(Chunk._meta.constraints, list)
        self.assertEqual(len(Chunk._meta.constraints), 2)
        self.assertTrue(
            all(isinstance(constraint, models.BaseConstraint) for constraint in Chunk._meta.constraints)
        )


class CreateRecordingTests(TestCase):
    def test_create_recording_persists_uploaded_recording(self) -> None:
        recording = create_recording(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/abc/interview.m4a",
            duration_milliseconds=125_000
        )

        self.assertEqual(Recording.objects.count(), 1)
        self.assertEqual(recording.original_file_name, "interview.m4a")
        self.assertEqual(
            recording.original_file_path, "data/recordings/abc/interview.m4a"
        )
        self.assertEqual(recording.duration_milliseconds, 125_000)
        self.assertEqual(recording.status, RecordingStatus.UPLOADED)

    def test_create_recording_strips_whitespace(self) -> None:
        recording = create_recording(
            original_file_name="  interview.m4a  ",
            original_file_path="  data/recordings/abc/original.m4a  ",
            duration_milliseconds=125_000,
        )
        self.assertEqual(recording.original_file_name, "interview.m4a")
        self.assertEqual(
            recording.original_file_path,
            "data/recordings/abc/original.m4a",
        )

    def test_create_recording_rejects_empty_filename(self) -> None:
        with self.assertRaisesMessage(ValueError, "original_file_name must not be empty"):
            create_recording(
                original_file_name="   ",
                original_file_path="data/recordings/abc/original.m4a",
                duration_milliseconds=125_000,
            )

    def test_create_recording_rejects_empty_path(self) -> None:
        with self.assertRaisesMessage(ValueError, "original_file_path must not be empty"):
            create_recording(
                original_file_name="interview.m4a",
                original_file_path="   ",
                duration_milliseconds=125_000,
            )

    def test_create_recording_rejects_non_positive_duration(self) -> None:
        with self.assertRaisesMessage(ValueError, "duration_milliseconds must be greater than 0"):
            create_recording(
                original_file_name="interview.m4a",
                original_file_path="data/recordings/abc/original.m4a",
                duration_milliseconds=0,
            )

    def test_create_recording_does_not_create_row_on_validation_error(self) -> None:
        with self.assertRaises(ValueError):
            create_recording(
                original_file_name="",
                original_file_path="data/recordings/abc/original.m4a",
                duration_milliseconds=125_000,
            )

        self.assertEqual(Recording.objects.count(), 0)


class NormalizeAudioTests(TestCase):
    def test_normalize_audio_updates_recording_on_success(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            recording_dir = Path(tmp_dir) / "recordings" / "abc"
            recording_dir.mkdir(parents=True, exist_ok=True)
            original_path = recording_dir / "original.m4a"
            original_path.write_bytes(b"fake-audio")
            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(original_path),
                duration_milliseconds=125_000,
                status=RecordingStatus.UPLOADED,
            )
            with patch("recordings.services._run_ffmpeg_normalization") as mock_ffmpeg:
                updated = normalize_audio(recording=recording)
            expected_normalized = recording_dir / "interview.m4a_normalized.wav"
            self.assertEqual(updated.normalized_file_path, str(expected_normalized))
            self.assertEqual(updated.status, RecordingStatus.NORMALIZED)
            recording.refresh_from_db()
            self.assertEqual(recording.normalized_file_path, str(expected_normalized))
            self.assertEqual(recording.status, RecordingStatus.NORMALIZED)
            mock_ffmpeg.assert_called_once_with(
                input_path=original_path,
                output_path=expected_normalized,
            )

    def test_normalize_audio_rejects_empty_original_path(self) -> None:
        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="",
            duration_milliseconds=125_000,
            status=RecordingStatus.UPLOADED,
        )
        with self.assertRaisesMessage(
                ValueError,
                "recording.original_file_path must not be empty",
        ):
            normalize_audio(recording=recording)

    def test_normalize_audio_raises_if_original_file_missing(self) -> None:
        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/missing/original.m4a",
            duration_milliseconds=125_000,
            status=RecordingStatus.UPLOADED,
        )
        with self.assertRaises(FileNotFoundError):
            normalize_audio(recording=recording)
        recording.refresh_from_db()
        self.assertEqual(recording.status, RecordingStatus.UPLOADED)
        self.assertIsNone(recording.normalized_file_path)

    def test_normalize_audio_does_not_update_recording_when_ffmpeg_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            recording_dir = Path(tmp_dir) / "recordings" / "abc"
            recording_dir.mkdir(parents=True, exist_ok=True)
            original_path = recording_dir / "original.m4a"
            original_path.write_bytes(b"fake-audio")
            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(original_path),
                duration_milliseconds=125_000,
                status=RecordingStatus.UPLOADED,
            )
            with patch(
                    "recordings.services._run_ffmpeg_normalization",
                    side_effect=AudioNormalizationError("ffmpeg failed"),
            ):
                with self.assertRaisesMessage(
                        AudioNormalizationError,
                        "ffmpeg failed",
                ):
                    normalize_audio(recording=recording)
            recording.refresh_from_db()
            self.assertEqual(recording.status, RecordingStatus.UPLOADED)
            self.assertIsNone(recording.normalized_file_path)


class RunFfmpegNormalizationTests(SimpleTestCase):
    @patch("recordings.services.subprocess.run")
    def test_run_ffmpeg_normalization_invokes_expected_command(self, mock_run) -> None:
        input_path = Path("/tmp/original.m4a")
        output_path = Path("/tmp/normalized.wav")
        _run_ffmpeg_normalization(
            input_path=input_path,
            output_path=output_path,
        )
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(
            args[0],
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
        )
        self.assertTrue(kwargs["check"])
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

    @patch("recordings.services.subprocess.run", side_effect=FileNotFoundError)
    def test_run_ffmpeg_normalization_raises_clear_error_if_ffmpeg_missing(self, mock_run) -> None:
        with self.assertRaisesMessage(
            AudioNormalizationError,
            "ffmpeg executable was not found",
        ):
            _run_ffmpeg_normalization(
                input_path=Path("/tmp/original.m4a"),
                output_path=Path("/tmp/normalized.wav"),
            )

    @patch(
        "recordings.services.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="invalid input",
        ),
    )
    def test_run_ffmpeg_normalization_raises_clear_error_on_ffmpeg_failure(self, mock_run) -> None:
        with self.assertRaisesMessage(
            AudioNormalizationError,
            "ffmpeg normalization failed: invalid input",
        ):
            _run_ffmpeg_normalization(
                input_path=Path("/tmp/original.m4a"),
                output_path=Path("/tmp/normalized.wav"),
            )