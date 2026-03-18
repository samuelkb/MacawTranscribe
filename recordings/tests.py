from django.db import models
from django.test import TestCase

from recordings.models import Chunk, Recording, RecordingStatus
from recordings.services import create_recording


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