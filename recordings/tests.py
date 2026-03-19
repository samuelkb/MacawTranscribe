import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
from uuid import uuid4

from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import models
from django.test import TestCase, SimpleTestCase, override_settings
from django.urls import reverse

from recordings.files import build_recording_directory, build_original_file_path, save_uploaded_file_atomic
from recordings.models import Chunk, Recording, RecordingStatus
from recordings.services import create_recording, normalize_audio, ingest_uploaded_recording
from recordings.audio import AudioNormalizationError, AudioProbeError, _run_ffmpeg_normalization, probe_audio_duration_milliseconds


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
            expected_normalized = recording_dir / "normalized.wav"
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


class ProbeAudioDurationMsTests(SimpleTestCase):
    def test_probe_audio_duration_ms_returns_integer_milliseconds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "audio.m4a"
            input_path.write_bytes(b"fake-audio")

            completed_process = Mock(stdout="12.345\n", stderr="")

            with patch(
                "recordings.audio.subprocess.run",
                return_value=completed_process,
            ) as mock_run:
                duration_ms = probe_audio_duration_milliseconds(input_path=input_path)

            self.assertEqual(duration_ms, 12_345)
            mock_run.assert_called_once()

    def test_probe_audio_duration_ms_rejects_blank_string_path(self) -> None:
        with self.assertRaisesMessage(ValueError, "input_path must not be empty"):
            probe_audio_duration_milliseconds(input_path=Path("   "))

    def test_probe_audio_duration_ms_raises_if_file_missing(self) -> None:
        missing_path = Path("/tmp/definitely_missing_audio_file.m4a")

        with self.assertRaises(FileNotFoundError):
            probe_audio_duration_milliseconds(input_path=missing_path)

    @patch("recordings.audio.subprocess.run", side_effect=FileNotFoundError)
    def test_probe_audio_duration_ms_raises_clear_error_if_ffprobe_missing(
        self,
        mock_run: Mock,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "audio.m4a"
            input_path.write_bytes(b"fake-audio")

            with self.assertRaisesMessage(
                AudioProbeError,
                "ffprobe executable was not found",
            ):
                probe_audio_duration_milliseconds(input_path=input_path)

    @patch(
        "recordings.audio.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffprobe"],
            stderr="invalid data found",
        ),
    )
    def test_probe_audio_duration_ms_raises_clear_error_on_ffprobe_failure(
        self,
        mock_run: Mock,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "audio.m4a"
            input_path.write_bytes(b"fake-audio")

            with self.assertRaisesMessage(
                AudioProbeError,
                "ffprobe failed: invalid data found",
            ):
                probe_audio_duration_milliseconds(input_path=input_path)

    def test_probe_audio_duration_ms_rejects_empty_stdout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "audio.m4a"
            input_path.write_bytes(b"fake-audio")

            completed_process = Mock(stdout="   \n", stderr="")

            with patch(
                "recordings.audio.subprocess.run",
                return_value=completed_process,
            ):
                with self.assertRaisesMessage(
                    AudioProbeError,
                    "ffprobe returned an empty duration",
                ):
                    probe_audio_duration_milliseconds(input_path=input_path)

    def test_probe_audio_duration_ms_rejects_non_numeric_stdout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "audio.m4a"
            input_path.write_bytes(b"fake-audio")

            completed_process = Mock(stdout="not-a-number\n", stderr="")

            with patch(
                "recordings.audio.subprocess.run",
                return_value=completed_process,
            ):
                with self.assertRaisesMessage(
                    AudioProbeError,
                    "ffprobe returned a non-numeric duration: 'not-a-number'",
                ):
                    probe_audio_duration_milliseconds(input_path=input_path)

    def test_probe_audio_duration_ms_rejects_non_positive_duration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "audio.m4a"
            input_path.write_bytes(b"fake-audio")

            completed_process = Mock(stdout="0\n", stderr="")

            with patch(
                "recordings.audio.subprocess.run",
                return_value=completed_process,
            ):
                with self.assertRaisesMessage(
                    AudioProbeError,
                    "ffprobe returned a non-positive duration: 0.0",
                ):
                    probe_audio_duration_milliseconds(input_path=input_path)


class RecordingFilesTest(SimpleTestCase):
    def test_build_recording_directory_uses_recording_id(self) -> None:
        recording_id = uuid4()
        base_dir = Path("data/recordings")

        result = build_recording_directory(
            recording_id=recording_id,
            base_dir=base_dir,
        )

        self.assertEqual(result, base_dir / str(recording_id))

    def test_build_original_file_path_preserves_extension(self) -> None:
        recording_id = uuid4()
        base_dir = Path("data/recordings")

        result = build_original_file_path(
            recording_id=recording_id,
            original_file_name="interview.m4a",
            base_dir=base_dir,
        )

        self.assertEqual(
            result,
            base_dir / str(recording_id) / "original.m4a",
        )

    def test_build_original_file_path_falls_back_to_bin_extension(self) -> None:
        recording_id = uuid4()
        base_dir = Path("data/recordings")

        result = build_original_file_path(
            recording_id=recording_id,
            original_file_name="interview",
            base_dir=base_dir,
        )

        self.assertEqual(
            result,
            base_dir / str(recording_id) / "original.bin",
        )

    def test_save_uploaded_file_atomic_writes_file_contents(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "recordings" / "abc" / "original.m4a"
            uploaded = SimpleUploadedFile(
                "interview.m4a",
                b"fake audio bytes",
                content_type="audio/mp4",
            )

            saved = save_uploaded_file_atomic(
                uploaded_file=uploaded,
                destination_path=destination,
            )

            self.assertEqual(saved, destination)
            self.assertTrue(destination.exists())
            self.assertEqual(destination.read_bytes(), b"fake audio bytes")

    def test_save_uploaded_file_atomic_rejects_none_upload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "recordings" / "abc" / "original.m4a"

            with self.assertRaisesMessage(ValueError, "uploaded_file cannot be None"):
                save_uploaded_file_atomic(
                    uploaded_file=None,  # type: ignore[arg-type]
                    destination_path=destination,
                )


class IngestUploadedRecordingTests(TestCase):
    def test_ingest_uploaded_recording_saves_file_and_creates_row(self) -> None:
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
                    recording = ingest_uploaded_recording(uploaded_file=uploaded)

            self.assertEqual(Recording.objects.count(), 1)
            self.assertEqual(recording.original_file_name, "interview.m4a")
            self.assertEqual(recording.duration_milliseconds, 125_000)
            self.assertEqual(recording.status, RecordingStatus.UPLOADED)

            original_path = Path(recording.original_file_path)
            self.assertTrue(original_path.exists())
            self.assertEqual(original_path.read_bytes(), b"fake audio bytes")
            self.assertEqual(original_path.name, "original.m4a")
            self.assertEqual(original_path.parent.name, str(recording.id))

    def test_ingest_uploaded_recording_rejects_none_upload(self) -> None:
        with self.assertRaisesMessage(ValueError, "uploaded_file must not be None"):
            ingest_uploaded_recording(uploaded_file=None)  # type: ignore[arg-type]


class UploadRecordingViewTests(TestCase):
    def test_upload_recording_returns_201_and_payload(self) -> None:
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
                    response = self.client.post(
                        reverse("recordings:upload_recording"),
                        {"file": uploaded},
                    )

        self.assertEqual(response.status_code, 201)

        payload = response.json()
        self.assertIn("recording_id", payload)
        self.assertEqual(payload["original_file_name"], "interview.m4a")
        self.assertEqual(payload["duration_milliseconds"], 125_000)
        self.assertEqual(payload["status"], RecordingStatus.UPLOADED)
        self.assertTrue(payload["original_file_path"].endswith("/original.m4a"))
        self.assertIsNone(payload["normalized_file_path"])

    def test_upload_recording_returns_400_when_file_is_missing(self) -> None:
        response = self.client.post(reverse("recordings:upload_recording"), {})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "missing_file")

    def test_upload_recording_returns_500_when_ingestion_fails(self) -> None:
        uploaded = SimpleUploadedFile(
            "interview.m4a",
            b"fake audio bytes",
            content_type="audio/mp4",
        )

        with patch(
                "recordings.views.ingest_uploaded_recording",
                side_effect=RuntimeError("boom"),
        ):
            response = self.client.post(
                reverse("recordings:upload_recording"),
                {"file": uploaded},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "uploaded_failed")
