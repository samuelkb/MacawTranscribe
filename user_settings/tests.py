from uuid import uuid4

from django.test import TestCase

from ml.types import BackendName, ModelName
from user_settings.models import WorkerRole, WorkerStatus, WorkerProcessState
from user_settings.services import register_worker_process, heartbeat_worker, mark_worker_idle, mark_worker_busy, \
    increment_jobs_processed, mark_worker_stopping, mark_worker_stopped, mark_worker_failed


class RegisterWorkerProcessTests(TestCase):
    def test_register_worker_process_creates_row(self) -> None:
        worker = register_worker_process(
            worker_id="worker-001",
            pid=12345,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            hostname="samuels-macbook",
        )

        self.assertEqual(worker.worker_id, "worker-001")
        self.assertEqual(worker.pid, 12345)
        self.assertEqual(worker.role, WorkerRole.TRANSCRIPTION)
        self.assertEqual(worker.status, WorkerStatus.STARTING)
        self.assertEqual(worker.backend, BackendName.MLX_WHISPER.value)
        self.assertEqual(worker.model, ModelName.MEDIUM.value)
        self.assertEqual(worker.hostname, "samuels-macbook")
        self.assertEqual(worker.jobs_processed, 0)
        self.assertIsNone(worker.current_chunk_id)
        self.assertEqual(worker.exit_reason, "")
        self.assertEqual(worker.last_error, "")
        self.assertIsNotNone(worker.started_at)
        self.assertIsNotNone(worker.last_heartbeat_at)
        self.assertIsNone(worker.stopped_at)

    def test_register_worker_process_updates_existing_row_by_worker_id(self) -> None:
        original = register_worker_process(
            worker_id="worker-001",
            pid=12345,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.SMALL,
            hostname="old-host",
        )

        updated = register_worker_process(
            worker_id="worker-001",
            pid=99999,
            role=WorkerRole.DIARIZATION,
            backend=None,
            model=None,
            hostname="new-host",
        )

        self.assertEqual(WorkerProcessState.objects.count(), 1)
        self.assertEqual(updated.id, original.id)
        self.assertEqual(updated.pid, 99999)
        self.assertEqual(updated.role, WorkerRole.DIARIZATION)
        self.assertIsNone(updated.backend)
        self.assertIsNone(updated.model)
        self.assertEqual(updated.hostname, "new-host")
        self.assertEqual(updated.status, WorkerStatus.STARTING)
        self.assertEqual(updated.jobs_processed, 0)
        self.assertEqual(updated.exit_reason, "")
        self.assertEqual(updated.last_error, "")
        self.assertIsNone(updated.stopped_at)

    def test_register_worker_process_allows_backend_and_model_to_be_none(self) -> None:
        worker = register_worker_process(
            worker_id="worker-002",
            pid=22222,
            role=WorkerRole.DIARIZATION,
            backend=None,
            model=None,
            hostname="host-2",
        )

        self.assertIsNone(worker.backend)
        self.assertIsNone(worker.model)


class WorkerHeartbeatTests(TestCase):
    def test_heartbeat_worker_updates_last_heartbeat(self) -> None:
        worker = register_worker_process(
            worker_id="worker-heartbeat",
            pid=20001,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.WHISPER_CPP,
            model=ModelName.LARGE_V3,
            hostname="host",
        )

        old_heartbeat = worker.last_heartbeat_at

        heartbeat_worker(worker_id=worker.worker_id)

        worker.refresh_from_db()
        self.assertGreaterEqual(worker.last_heartbeat_at, old_heartbeat)


class MarkWorkerIdleTests(TestCase):
    def test_mark_worker_idle_sets_idle_status_and_clears_current_chunk(self) -> None:
        chunk_id = uuid4()
        worker = register_worker_process(
            worker_id="worker-idle",
            pid=20002,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.SMALL,
            hostname="host",
        )
        WorkerProcessState.objects.filter(worker_id=worker.worker_id).update(
            status=WorkerStatus.BUSY,
            current_chunk_id=chunk_id,
        )

        mark_worker_idle(worker_id=worker.worker_id)

        worker.refresh_from_db()
        self.assertEqual(worker.status, WorkerStatus.IDLE)
        self.assertIsNone(worker.current_chunk_id)


class MarkWorkerBusyTests(TestCase):
    def test_mark_worker_busy_sets_busy_status_and_chunk_id(self) -> None:
        chunk_id = uuid4()
        worker = register_worker_process(
            worker_id="worker-busy",
            pid=20003,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            hostname="host",
        )

        mark_worker_busy(worker_id=worker.worker_id, chunk_id=chunk_id)

        worker.refresh_from_db()
        self.assertEqual(worker.status, WorkerStatus.BUSY)
        self.assertEqual(worker.current_chunk_id, chunk_id)


class IncrementJobsProcessedTests(TestCase):
    def test_increment_jobs_processed_increments_counter(self) -> None:
        worker = register_worker_process(
            worker_id="worker-jobs",
            pid=20005,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.LARGE_V3,
            hostname="host",
        )

        self.assertEqual(worker.jobs_processed, 0)

        increment_jobs_processed(worker_id=worker.worker_id)
        worker.refresh_from_db()
        self.assertEqual(worker.jobs_processed, 1)

        increment_jobs_processed(worker_id=worker.worker_id)
        worker.refresh_from_db()
        self.assertEqual(worker.jobs_processed, 2)

    def test_increment_jobs_processed_updates_last_heartbeat(self) -> None:
        worker = register_worker_process(
            worker_id="worker-jobs-heartbeat",
            pid=20006,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            hostname="host",
        )

        old_heartbeat = worker.last_heartbeat_at

        increment_jobs_processed(worker_id=worker.worker_id)

        worker.refresh_from_db()
        self.assertGreaterEqual(worker.last_heartbeat_at, old_heartbeat)


class MarkWorkerStoppingTests(TestCase):
    def test_mark_worker_stopping_sets_status_and_exit_reason(self) -> None:
        worker = register_worker_process(
            worker_id="worker-stopping",
            pid=20007,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.WHISPER_CPP,
            model=ModelName.SMALL,
            hostname="host",
        )

        mark_worker_stopping(
            worker_id=worker.worker_id,
            exit_reason="max_jobs_reached",
        )

        worker.refresh_from_db()
        self.assertEqual(worker.status, WorkerStatus.STOPPING)
        self.assertEqual(worker.exit_reason, "max_jobs_reached")


class MarkWorkerStoppedTests(TestCase):
    def test_mark_worker_stopped_sets_status_and_stopped_at(self) -> None:
        chunk_id = uuid4()
        worker = register_worker_process(
            worker_id="worker-stopped",
            pid=20008,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.MLX_WHISPER,
            model=ModelName.MEDIUM,
            hostname="host",
        )
        WorkerProcessState.objects.filter(worker_id=worker.worker_id).update(
            status=WorkerStatus.BUSY,
            current_chunk_id=chunk_id,
        )

        mark_worker_stopped(
            worker_id=worker.worker_id,
            exit_reason="graceful_shutdown",
        )

        worker.refresh_from_db()
        self.assertEqual(worker.status, WorkerStatus.STOPPED)
        self.assertEqual(worker.exit_reason, "graceful_shutdown")
        self.assertIsNone(worker.current_chunk_id)
        self.assertIsNotNone(worker.stopped_at)


class MarkWorkerFailedTests(TestCase):
    def test_mark_worker_failed_sets_failed_status_and_error(self) -> None:
        worker = register_worker_process(
            worker_id="worker-failed",
            pid=20009,
            role=WorkerRole.TRANSCRIPTION,
            backend=BackendName.WHISPER_CPP,
            model=ModelName.LARGE_V3,
            hostname="host",
        )

        mark_worker_failed(
            worker_id=worker.worker_id,
            error_message="worker crashed during inference",
        )

        worker.refresh_from_db()
        self.assertEqual(worker.status, WorkerStatus.FAILED)
        self.assertEqual(worker.last_error, "worker crashed during inference")
        self.assertIsNotNone(worker.stopped_at)