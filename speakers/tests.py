from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.db import IntegrityError, transaction
from django.test import TestCase

from recordings.models import Recording, RecordingStatus
from speakers.audio import DiarizationSegment, DiarizationError, VadSpeechSegment
from speakers.models import SpeakerSegment, SilenceSegment, SpeakerLabel
from speakers.services import run_diarization, derive_silence_intervals, run_vad


class SpeakerModelsTests(TestCase):
    def setUp(self) -> None:
        self.recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/test-recording/original.m4a",
            duration_milliseconds=120_000,
            status=RecordingStatus.UPLOADED,
        )

    def test_speaker_segment_duration_ms_property(self) -> None:
        segment = SpeakerSegment.objects.create(
            recording=self.recording,
            speaker_id="SPEAKER_00",
            start_time=1_000,
            end_time=4_500,
            model_used="pyannote",
        )

        self.assertEqual(segment.duration_milliseconds, 3_500)

    def test_silence_segment_duration_ms_property(self) -> None:
        segment = SilenceSegment.objects.create(
            recording=self.recording,
            start_time=10_000,
            end_time=12_250,
            model_used="silero_vad",
        )

        self.assertEqual(segment.duration_milliseconds, 2_250)

    def test_speaker_label_must_be_unique_per_recording_and_speaker_id(self) -> None:
        SpeakerLabel.objects.create(
            recording=self.recording,
            speaker_id="SPEAKER_00",
            display_name="Interviewer",
        )

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SpeakerLabel.objects.create(
                    recording=self.recording,
                    speaker_id="SPEAKER_00",
                    display_name="Host",
                )

    def test_speaker_label_same_speaker_id_can_exist_for_different_recordings(self) -> None:
        other_recording = Recording.objects.create(
            original_file_name="other.m4a",
            original_file_path="data/recordings/other-recording/original.m4a",
            duration_milliseconds=90_000,
            status=RecordingStatus.UPLOADED,
        )

        SpeakerLabel.objects.create(
            recording=self.recording,
            speaker_id="SPEAKER_00",
            display_name="Interviewer",
        )

        SpeakerLabel.objects.create(
            recording=other_recording,
            speaker_id="SPEAKER_00",
            display_name="Guest",
        )

        self.assertEqual(SpeakerLabel.objects.count(), 2)

    def test_speaker_segment_rejects_end_ms_less_than_start_ms(self) -> None:
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SpeakerSegment.objects.create(
                    recording=self.recording,
                    speaker_id="SPEAKER_00",
                    start_time=5_000,
                    end_time=4_999,
                    model_used="pyannote",
                )

    def test_speaker_segment_rejects_equal_start_and_end_ms(self) -> None:
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SpeakerSegment.objects.create(
                    recording=self.recording,
                    speaker_id="SPEAKER_00",
                    start_time=5_000,
                    end_time=5_000,
                    model_used="pyannote",
                )

    def test_silence_segment_rejects_end_ms_less_than_start_ms(self) -> None:
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SilenceSegment.objects.create(
                    recording=self.recording,
                    start_time=8_000,
                    end_time=7_999,
                    model_used="silero_vad",
                )

    def test_silence_segment_rejects_equal_start_and_end_ms(self) -> None:
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                SilenceSegment.objects.create(
                    recording=self.recording,
                    start_time=8_000,
                    end_time=8_000,
                    model_used="silero_vad",
                )

    def test_speaker_segment_string_representation(self) -> None:
        segment = SpeakerSegment.objects.create(
            recording=self.recording,
            speaker_id="SPEAKER_00",
            start_time=1_000,
            end_time=4_500,
            model_used="pyannote",
        )

        self.assertIn("SPEAKER_00", str(segment))
        self.assertIn(str(self.recording.id), str(segment))

    def test_silence_segment_string_representation(self) -> None:
        segment = SilenceSegment.objects.create(
            recording=self.recording,
            start_time=10_000,
            end_time=12_250,
            model_used="silero_vad",
        )

        self.assertIn("silence", str(segment))
        self.assertIn(str(self.recording.id), str(segment))

    def test_speaker_label_string_representation(self) -> None:
        label = SpeakerLabel.objects.create(
            recording=self.recording,
            speaker_id="SPEAKER_00",
            display_name="Interviewer",
        )

        self.assertIn("SPEAKER_00", str(label))
        self.assertIn("Interviewer", str(label))


class RunDiarizationTests(TestCase):
    def test_run_diarization_persists_segments_and_updates_recording_status(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            normalized_path = Path(tmp_dir) / "normalized.wav"
            normalized_path.write_bytes(b"fake-normalized-audio")

            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(Path(tmp_dir) / "original.m4a"),
                normalized_file_path=str(normalized_path),
                duration_milliseconds=120_000,
                status=RecordingStatus.NORMALIZED,
            )

            fake_segments = [
                DiarizationSegment(
                    speaker_id="SPEAKER_00",
                    start_time=0,
                    end_time=5_000,
                ),
                DiarizationSegment(
                    speaker_id="SPEAKER_01",
                    start_time=5_000,
                    end_time=10_000,
                ),
            ]

            with patch(
                    "speakers.services.run_pyannote_diarization",
                    return_value=fake_segments,
            ):
                with patch(
                        "speakers.services.get_diarization_model_name",
                        return_value="pyannote/speaker-diarization-community-1",
                ):
                    created = run_diarization(recording=recording)

            self.assertEqual(len(created), 2)
            self.assertEqual(SpeakerSegment.objects.count(), 2)

            recording.refresh_from_db()
            self.assertEqual(recording.status, RecordingStatus.DIARIZED)

            first = SpeakerSegment.objects.order_by("start_time").first()
            assert first is not None
            self.assertEqual(first.speaker_id, "SPEAKER_00")
            self.assertEqual(first.start_time, 0)
            self.assertEqual(first.end_time, 5_000)
            self.assertEqual(
                first.model_used,
                "pyannote/speaker-diarization-community-1",
            )

    def test_run_diarization_replaces_existing_segments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            normalized_path = Path(tmp_dir) / "normalized.wav"
            normalized_path.write_bytes(b"fake-normalized-audio")

            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(Path(tmp_dir) / "original.m4a"),
                normalized_file_path=str(normalized_path),
                duration_milliseconds=120_000,
                status=RecordingStatus.NORMALIZED,
            )

            SpeakerSegment.objects.create(
                recording=recording,
                speaker_id="OLD_SPEAKER",
                start_time=0,
                end_time=1_000,
                model_used="old-model",
            )

            fake_segments = [
                DiarizationSegment(
                    speaker_id="SPEAKER_00",
                    start_time=2_000,
                    end_time=6_000,
                ),
            ]

            with patch(
                    "speakers.services.run_pyannote_diarization",
                    return_value=fake_segments,
            ):
                with patch(
                        "speakers.services.get_diarization_model_name",
                        return_value="pyannote/speaker-diarization-community-1",
                ):
                    created = run_diarization(recording=recording)

            self.assertEqual(SpeakerSegment.objects.count(), 1)
            segment = SpeakerSegment.objects.get()
            self.assertEqual(segment.speaker_id, "SPEAKER_00")
            self.assertEqual(segment.start_time, 2_000)
            self.assertEqual(segment.end_time, 6_000)

    def test_run_diarization_rejects_missing_normalized_file_path(self) -> None:
        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/x/original.m4a",
            duration_milliseconds=120_000,
            status=RecordingStatus.UPLOADED,
        )

        with self.assertRaisesMessage(
                ValueError,
                "recording.normalized_file_path must not be empty",
        ):
            run_diarization(recording=recording)

    def test_run_diarization_propagates_backend_errors_without_modifying_status(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            normalized_path = Path(tmp_dir) / "normalized.wav"
            normalized_path.write_bytes(b"fake-normalized-audio")

            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(Path(tmp_dir) / "original.m4a"),
                normalized_file_path=str(normalized_path),
                duration_milliseconds=120_000,
                status=RecordingStatus.NORMALIZED,
            )

            with patch(
                    "speakers.services.run_pyannote_diarization",
                    side_effect=DiarizationError("boom"),
            ):
                with self.assertRaisesMessage(DiarizationError, "boom"):
                    run_diarization(recording=recording)

            recording.refresh_from_db()
            self.assertEqual(recording.status, RecordingStatus.NORMALIZED)
            self.assertEqual(SpeakerSegment.objects.count(), 0)


class DeriveSilenceIntervalsTest(TestCase):
    def test_returns_full_duration_as_silence_when_no_speech_segments(self) -> None:
        intervals = derive_silence_intervals(
            speech_segments=[],
            duration_milliseconds=10_000,
        )

        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0].start_time, 0)
        self.assertEqual(intervals[0].end_time, 10_000)

    def test_derives_gaps_before_between_and_after_speech(self) -> None:
        intervals = derive_silence_intervals(
            speech_segments=[
                VadSpeechSegment(start_time=1_000, end_time=2_000),
                VadSpeechSegment(start_time=4_000, end_time=5_000),
            ],
            duration_milliseconds=7_000,
        )

        self.assertEqual(
            [(i.start_time, i.end_time) for i in intervals],
            [
                (0, 1_000),
                (2_000, 4_000),
                (5_000, 7_000),
            ],
        )

    def test_handles_unsorted_speech_segments(self) -> None:
        intervals = derive_silence_intervals(
            speech_segments=[
                VadSpeechSegment(start_time=4_000, end_time=5_000),
                VadSpeechSegment(start_time=1_000, end_time=2_000),
            ],
            duration_milliseconds=7_000,
        )

        self.assertEqual(
            [(i.start_time, i.end_time) for i in intervals],
            [
                (0, 1_000),
                (2_000, 4_000),
                (5_000, 7_000),
            ],
        )

    def test_rejects_non_positive_duration(self) -> None:
        with self.assertRaisesMessage(ValueError, "duration_milliseconds must be greater than 0"):
            derive_silence_intervals(
                speech_segments=[],
                duration_milliseconds=0,
            )


class RunVadTest(TestCase):
    def test_run_vad_persists_silence_segments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            normalized_path = Path(tmp_dir) / "normalized.wav"
            normalized_path.write_bytes(b"fake-normalized-audio")

            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(Path(tmp_dir) / "original.m4a"),
                normalized_file_path=str(normalized_path),
                duration_milliseconds=10_000,
                status=RecordingStatus.DIARIZED,
            )

            fake_speech_segments = [
                VadSpeechSegment(start_time=1_000, end_time=2_000),
                VadSpeechSegment(start_time=4_000, end_time=5_000),
            ]

            with patch(
                    "speakers.services.run_vad_backend",
                    return_value=fake_speech_segments,
            ):
                with patch(
                        "speakers.services.get_vad_model_name",
                        return_value="silero_vad",
                ):
                    created = run_vad(recording=recording)

            self.assertEqual(len(created), 3)
            self.assertEqual(SilenceSegment.objects.count(), 3)

            intervals = list(
                SilenceSegment.objects.order_by("start_time").values_list(
                    "start_time",
                    "end_time",
                    "model_used",
                )
            )
            self.assertEqual(
                intervals,
                [
                    (0, 1_000, "silero_vad"),
                    (2_000, 4_000, "silero_vad"),
                    (5_000, 10_000, "silero_vad"),
                ],
            )

    def test_run_vad_replaces_existing_silence_segments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            normalized_path = Path(tmp_dir) / "normalized.wav"
            normalized_path.write_bytes(b"fake-normalized-audio")

            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(Path(tmp_dir) / "original.m4a"),
                normalized_file_path=str(normalized_path),
                duration_milliseconds=10_000,
                status=RecordingStatus.DIARIZED,
            )

            SilenceSegment.objects.create(
                recording=recording,
                start_time=0,
                end_time=500,
                model_used="old_vad",
            )

            with patch(
                    "speakers.services.run_vad_backend",
                    return_value=[VadSpeechSegment(start_time=2_000, end_time=3_000)],
            ):
                with patch(
                        "speakers.services.get_vad_model_name",
                        return_value="silero_vad",
                ):
                    run_vad(recording=recording)

            intervals = list(
                SilenceSegment.objects.order_by("start_time").values_list(
                    "start_time",
                    "end_time",
                )
            )
            self.assertEqual(intervals, [(0, 2_000), (3_000, 10_000)])

    def test_run_vad_rejects_missing_normalized_file_path(self) -> None:
        recording = Recording.objects.create(
            original_file_name="interview.m4a",
            original_file_path="data/recordings/x/original.m4a",
            duration_milliseconds=10_000,
            status=RecordingStatus.UPLOADED,
        )

        with self.assertRaisesMessage(
                ValueError,
                "recording.normalized_file_path must not be empty",
        ):
            run_vad(recording=recording)

    def test_run_vad_logs_start_and_completion(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            normalized_path = Path(tmp_dir) / "normalized.wav"
            normalized_path.write_bytes(b"fake-normalized-audio")

            recording = Recording.objects.create(
                original_file_name="interview.m4a",
                original_file_path=str(Path(tmp_dir) / "original.m4a"),
                normalized_file_path=str(normalized_path),
                duration_milliseconds=10_000,
                status=RecordingStatus.DIARIZED,
            )

            with patch(
                    "speakers.services.run_vad_backend",
                    return_value=[VadSpeechSegment(start_time=2_000, end_time=3_000)],
            ):
                with patch(
                        "speakers.services.get_vad_model_name",
                        return_value="silero_vad",
                ):
                    with self.assertLogs("speakers.services", level="INFO") as captured:
                        run_vad(recording=recording)

            output = "\n".join(captured.output)
            self.assertIn("recording_vad_started", output)
            self.assertIn("recording_vad_completed", output)

