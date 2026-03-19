from django.db import IntegrityError, transaction
from django.test import TestCase

from recordings.models import Recording, RecordingStatus
from speakers.models import SpeakerSegment, SilenceSegment, SpeakerLabel


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