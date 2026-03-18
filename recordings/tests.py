from django.db import models
from django.test import TestCase

from recordings.models import Chunk


class ChunkMetaTests(TestCase):
    def test_constraints_are_constraint_objects(self) -> None:
        self.assertIsInstance(Chunk._meta.constraints, list)
        self.assertEqual(len(Chunk._meta.constraints), 2)
        self.assertTrue(
            all(isinstance(constraint, models.BaseConstraint) for constraint in Chunk._meta.constraints)
        )
