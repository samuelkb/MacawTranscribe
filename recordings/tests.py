import json
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
from recordings.models import Chunk, Recording, RecordingStatus, ChunkStatus
from recordings.services import (
    create_recording,
    normalize_audio,
    ingest_uploaded_recording,
    create_chunks,
    format_duration_hhmmss,
)
from recordings.audio import AudioNormalizationError, AudioProbeError, _run_ffmpeg_normalization, \
    probe_audio_duration_milliseconds, extract_chunk_audio, ChunkAudioExtractionError


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


class RecordingDurationFormattingTests(SimpleTestCase):
    def test_format_duration_hhmmss_renders_expected_output(self) -> None:
        self.assertEqual(format_duration_hhmmss(duration_milliseconds=125_000), "00:02:05")

    def test_format_duration_hhmmss_handles_missing_value(self) -> None:
        self.assertEqual(format_duration_hhmmss(duration_milliseconds=None), "00:00:00")


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


class CreateChunksTests(TestCase):
    def setUp(self) -> None:
        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=80_000,
            status=RecordingStatus.DIARIZED,
        )

    def test_create_chunks_generates_fixed_chunks_with_overlap(self) -> None:
        chunks = create_chunks(
            recording=self.recording,
            chunk_duration_milliseconds=30_000,
            overlap_milliseconds=5_000,
        )

        self.assertEqual(len(chunks), 3)

        persisted = list(
            Chunk.objects.filter(recording=self.recording)
            .order_by("chunk_index")
            .values_list("chunk_index", "start_time", "end_time", "status")
        )

        self.assertEqual(
            persisted,
            [
                (0, 0, 30_000, ChunkStatus.PENDING),
                (1, 25_000, 55_000, ChunkStatus.PENDING),
                (2, 50_000, 80_000, ChunkStatus.PENDING),
            ],
        )

        self.recording.refresh_from_db()
        self.assertEqual(self.recording.status, RecordingStatus.CHUNKED)

    def test_create_chunks_creates_single_chunk_for_short_recording(self) -> None:
        self.recording.duration_milliseconds = 10_000
        self.recording.save(update_fields=["duration_milliseconds"])

        chunks = create_chunks(
            recording=self.recording,
            chunk_duration_milliseconds=30_000,
            overlap_milliseconds=5_000,
        )

        self.assertEqual(len(chunks), 1)

        chunk = Chunk.objects.get(recording=self.recording)
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.start_time, 0)
        self.assertEqual(chunk.end_time, 10_000)

    def test_create_chunks_replaces_existing_chunks(self) -> None:
        Chunk.objects.create(
            recording=self.recording,
            chunk_index=0,
            start_time=0,
            end_time=10_000,
            status=ChunkStatus.COMPLETED,
        )

        create_chunks(
            recording=self.recording,
            chunk_duration_milliseconds=30_000,
            overlap_milliseconds=5_000,
        )

        persisted = list(
            Chunk.objects.filter(recording=self.recording)
            .order_by("chunk_index")
            .values_list("chunk_index", "start_time", "end_time")
        )

        self.assertEqual(
            persisted,
            [
                (0, 0, 30_000),
                (1, 25_000, 55_000),
                (2, 50_000, 80_000),
            ],
        )

    def test_create_chunks_rejects_non_positive_duration(self) -> None:
        self.recording.duration_milliseconds = 0
        self.recording.save(update_fields=["duration_milliseconds"])

        with self.assertRaisesMessage(
                ValueError,
                "recording.duration_milliseconds must be greater than 0",
        ):
            create_chunks(recording=self.recording)

    def test_create_chunks_rejects_non_positive_chunk_duration(self) -> None:
        with self.assertRaisesMessage(
                ValueError,
                "chunk_duration_milliseconds must be greater than 0",
        ):
            create_chunks(
                recording=self.recording,
                chunk_duration_milliseconds=0,
            )

    def test_create_chunks_rejects_negative_overlap(self) -> None:
        with self.assertRaisesMessage(
                ValueError,
                "overlap_milliseconds must not be negative",
        ):
            create_chunks(
                recording=self.recording,
                overlap_milliseconds=-1,
            )

    def test_create_chunks_rejects_overlap_greater_than_or_equal_to_chunk_duration(self) -> None:
        with self.assertRaisesMessage(
                ValueError,
                "overlap_milliseconds must be smaller than chunk_duration_milliseconds",
        ):
            create_chunks(
                recording=self.recording,
                chunk_duration_milliseconds=30_000,
                overlap_milliseconds=30_000,
            )

    def test_create_chunks_logs_start_and_completion(self) -> None:
        with self.assertLogs("recordings.services", level="INFO") as captured:
            create_chunks(
                recording=self.recording,
                chunk_duration_milliseconds=30_000,
                overlap_milliseconds=5_000,
            )

        output = "\n".join(captured.output)
        self.assertIn("chunk_creation_started", output)
        self.assertIn("chunk_creation_completed", output)


class CreateChunksViewTest(TestCase):
    def setUp(self) -> None:
        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test/original.m4a",
            normalized_file_path="data/recordings/test/normalized.wav",
            duration_milliseconds=80_000,
            status=RecordingStatus.DIARIZED,
        )

    def test_returns_201_and_payload_on_success(self) -> None:
        with patch("recordings.views.create_chunks", return_value=[object(), object(), object()]):
            response = self.client.post(
                reverse("recordings:create_chunks", args=[self.recording.id]),
                data=json.dumps({"chunk_duration_milliseconds": 30000, "overlap_milliseconds": 5000}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["recording_id"], str(self.recording.id))
        self.assertEqual(payload["chunk_count"], 3)
        self.assertEqual(payload["chunk_duration_milliseconds"], 30000)
        self.assertEqual(payload["overlap_milliseconds"], 5000)

    def test_returns_404_when_recording_missing(self) -> None:
        response = self.client.post(
            reverse("recordings:create_chunks", args=[uuid4()]),
            data="{}",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 404)

    def test_returns_400_on_invalid_json(self) -> None:
        response = self.client.post(
            reverse("recordings:create_chunks", args=[self.recording.id]),
            data="{bad json",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    def test_returns_400_when_service_rejects_request(self) -> None:
        with patch(
                "recordings.views.create_chunks",
                side_effect=ValueError("recording.normalized_file_path must not be empty"),
        ):
            response = self.client.post(
                reverse("recordings:create_chunks", args=[self.recording.id]),
                data="{}",
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 400)

    def test_returns_500_on_unexpected_failure(self) -> None:
        with patch("recordings.views.create_chunks", side_effect=RuntimeError("boom")):
            response = self.client.post(
                reverse("recordings:create_chunks", args=[self.recording.id]),
                data="{}",
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 500)


class ExtractChunkAudioTests(TestCase):
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
            start_time=5_000,
            end_time=20_000,
            status=ChunkStatus.PENDING,
        )

    @patch("recordings.audio.subprocess.run")
    def test_extract_chunk_audio_returns_temp_wav_path(self, mock_run: Mock) -> None:
        output_path = extract_chunk_audio(chunk=self.chunk)

        self.assertTrue(output_path.name.endswith(".wav"))
        self.assertTrue(output_path.exists())

        args, kwargs = mock_run.call_args
        command = args[0]

        self.assertIn(str(self.normalized_path), command)
        self.assertIn(str(output_path), command)
        self.assertTrue(kwargs["check"])
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

        output_path.unlink(missing_ok=True)

    def test_extract_chunk_audio_rejects_negative_start(self) -> None:
        self.chunk.start_time = -1
        self.chunk.save(update_fields=["start_time"])

        with self.assertRaisesMessage(
                ValueError,
                "chunk.start_time must not be negative",
        ):
            extract_chunk_audio(chunk=self.chunk)

    def test_extract_chunk_audio_raises_if_source_missing(self) -> None:
        self.normalized_path.unlink()

        with self.assertRaises(FileNotFoundError):
            extract_chunk_audio(chunk=self.chunk)

    @patch("recordings.audio.subprocess.run", side_effect=FileNotFoundError)
    def test_extract_chunk_audio_raises_clear_error_if_ffmpeg_missing(self, mock_run: Mock) -> None:
        with self.assertRaisesMessage(
                ChunkAudioExtractionError,
                "ffmpeg executable was not found",
        ):
            extract_chunk_audio(chunk=self.chunk)

    @patch(
        "recordings.audio.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="bad input",
        ),
    )
    def test_extract_chunk_audio_raises_clear_error_on_ffmpeg_failure(self, mock_run: Mock) -> None:
        with self.assertRaisesMessage(
                ChunkAudioExtractionError,
                "ffmpeg chunk extraction failed: bad input",
        ):
            extract_chunk_audio(chunk=self.chunk)

    @patch("recordings.audio.subprocess.run")
    def test_extract_chunk_audio_logs_start_and_completion(self, mock_run: Mock) -> None:
        with self.assertLogs("recordings.audio", level="INFO") as captured:
            output_path = extract_chunk_audio(chunk=self.chunk)

        output = "\n".join(captured.output)
        self.assertIn("chunk_audio_extraction_started", output)
        self.assertIn("chunk_audio_extraction_completed", output)

        output_path.unlink(missing_ok=True)
