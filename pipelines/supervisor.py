import logging
import multiprocessing
import os
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.synchronize import Event as MpEvent
from typing import Final, TYPE_CHECKING
from uuid import uuid4

logger: Final[logging.Logger] = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pipelines.worker import WorkerConfig
    from user_settings.models import TranscriptionRuntimeSettings, WorkerProcessState


@dataclass
class ManagedWorker:
    worker_id: str
    process: multiprocessing.Process
    stop_event: MpEvent
    started_at: float
    job_family: str
    partition_key: str
    queue_name: str


def _worker_entrypoint(*, worker_id: str, stop_event: MpEvent, config_dict: dict) -> None:
    """
    Child process entrypoint.

    This wraps the existing worker loop and ensures fatal failures are recorded.
    :param worker_id: Worker identifier.
    :param stop_event: Event to stop worker loop.
    :param config: Worker configuration.
    """
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pipelines.settings")
        import django
        django.setup()
        from pipelines.worker import WorkerConfig, run_worker_loop
        config = WorkerConfig(**config_dict)
        run_worker_loop(worker_id=worker_id, config=config, stop_event=stop_event)
    except Exception as exc:
        logger.exception(
            "worker_process_fatal_failure",
            extra={
                "worker_id": worker_id,
                "error": str(exc),
            }
        )
        try:
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pipelines.settings")
            import django
            django.setup()
            from user_settings.services import mark_worker_failed
            mark_worker_failed(worker_id=worker_id, error_message=str(exc))
        except Exception:
            logger.exception("worker_process_failed_to_persist_failure_state", extra={"worker_id": worker_id})
        raise


class WorkerSupervisor:
    def __init__(self) -> None:
        self._workers: dict[str, ManagedWorker] = {}
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run_forever(self) -> None:
        from django.db import close_old_connections
        from user_settings.models import TranscriptionRuntimeSettings
        logger.info("worker_supervisor_started")

        try:
            while not self._stop_requested:
                close_old_connections()
                settings = TranscriptionRuntimeSettings.get_solo()
                self._reap_dead_workers()
                self._reconcile(settings=settings)
                close_old_connections()
                time.sleep(settings.supervisor_poll_seconds)
        finally:
            close_old_connections()
            self._shutdown_all_workers()
            close_old_connections()
            logger.info("worker_supervisor_stopped")

    def _reconcile(self, *, settings: TranscriptionRuntimeSettings) -> None:
        if not settings.supervisor_enabled:
            logger.info("worker_supervisor_disabled_stopping_all_workers")
            self._scale_down_to_zero()
            return

        self._recycle_eligible_workers(settings=settings)
        target_count = min(settings.desired_worker_count, settings.max_worker_count)
        current_count = len(self._workers)

        logger.info(
            "worker_supervisor_reconcile",
            extra={
                "supervisor_enabled": settings.supervisor_enabled,
                "desired_worker_count": settings.desired_worker_count,
                "max_worker_count": settings.max_worker_count,
                "target_count": target_count,
                "current_count": current_count,
            }
        )

        if current_count < target_count:
            missing = target_count - current_count
            for _ in range(missing):
                self._spawn_worker()
        elif current_count > target_count:
            extra = current_count - target_count
            self._stop_extra_workers(extra_count=extra)

    def _spawn_worker(self) -> None:
        from pipelines.worker import WorkerConfig
        from user_settings.services import get_default_worker_backend_and_model
        from pipelines.queue_names import build_transcription_partition_key, build_transcription_queue_name

        backend, model = get_default_worker_backend_and_model()
        if backend is None or model is None:
            raise RuntimeError("Default transcription backend/model must be configured before spawning workers.")
        partition_key = build_transcription_partition_key(backend=backend, model=model)
        queue_name = build_transcription_queue_name(backend=backend, model=model)
        worker_id = f"{socket.gethostname()}:{uuid4().hex[:12]}"
        stop_event = multiprocessing.Event()
        config = WorkerConfig()
        config_dict = {
            "heartbeat_interval_seconds": config.heartbeat_interval_seconds,
            "stale_after_seconds": config.stale_after_seconds,
            "dequeue_timeout_seconds": config.dequeue_timeout_seconds,
            "recover_stale_chunks_on_startup": config.recover_stale_chunks_on_startup,
        }
        process = multiprocessing.Process(
            target=_worker_entrypoint,
            kwargs={
                "worker_id": worker_id,
                "stop_event": stop_event,
                "config_dict": config_dict,
            },
            daemon=False,
            name=f"transcription-worker-process-{worker_id}",
        )
        process.start()
        self._workers[worker_id] = ManagedWorker(
            worker_id=worker_id,
            process=process,
            stop_event=stop_event,
            started_at=time.time(),
            job_family="transcription",
            partition_key=partition_key,
            queue_name=queue_name,
        )
        logger.info(
            "worker_supervisor_spawned_worker",
            extra={
                "worker_id": worker_id,
                "pid": process.pid,
                "job_family": "transcription",
                "partition_key": partition_key,
                "queue_name": queue_name,
            }
        )

    def _stop_extra_workers(self, *, extra_count: int) -> None:
        worker_ids = list(self._workers.keys())[:extra_count]
        for worker_id in worker_ids:
            self._stop_worker(worker_id=worker_id, reason="scale_down")

    def _scale_down_to_zero(self) -> None:
        for worker_id in list(self._workers.keys()):
            self._stop_worker(worker_id=worker_id, reason="supervisor_disabled")

    def _stop_worker(self, *, worker_id: str, reason: str) -> None:
        from user_settings.services import mark_worker_stopped
        managed = self._workers.get(worker_id)
        if managed is None:
            return
        logger.info(
            "worker_supervisor_stopping_worker",
            extra={
                "worker_id": worker_id,
                "pid": managed.process.pid,
                "reason": reason,
            }
        )
        managed.stop_event.set()
        managed.process.join(timeout=10)
        if managed.process.is_alive():
            logger.warning(
                "worker_supervisor_terminating_worker",
                extra={
                    "worker_id": worker_id,
                    "pid": managed.process.pid,
                    "reason": reason,
                }
            )
            managed.process.terminate()
            managed.process.join(timeout=5)
        try:
            mark_worker_stopped(
                worker_id=worker_id,
                exit_reason=reason,
            )
        except Exception:
            logger.exception(
                "worker_supervisor_failed_to_mark_worker_stopped",
                extra={
                    "worker_id": worker_id,
                }
            )
        self._workers.pop(worker_id, None)

    def _reap_dead_workers(self) -> None:
        from user_settings.services import mark_worker_stopped, mark_worker_failed
        dead_worker_ids: list[str] = []
        for worker_id, managed in self._workers.items():
            if managed.process.is_alive():
                continue
            exitcode = managed.process.exitcode
            logger.warning(
                "worker_supervisor_detected_dead_worker",
                extra={
                    "worker_id": worker_id,
                    "pid": managed.process.pid,
                    "exitcode": exitcode,
                }
            )
            try:
                if exitcode == 0:
                    mark_worker_stopped(worker_id=worker_id, exit_reason="exited_normally")
                else:
                    mark_worker_failed(
                        worker_id=worker_id,
                        error_message=f"worker exited unexpectedly with code {exitcode}",
                    )
            except Exception:
                logger.exception(
                    "worker_supervisor_failed_to_record_dead_worker",
                    extra={"worker_id": worker_id},
                )
            dead_worker_ids.append(worker_id)

        for worker_id in dead_worker_ids:
            self._workers.pop(worker_id, None)

    def _shutdown_all_workers(self) -> None:
        for worker_id in list(self._workers.keys()):
            self._stop_worker(worker_id=worker_id, reason="supervisor_shutdown")

    def _has_worker_reached_max_jobs(self, *, worker_state: WorkerProcessState, settings: TranscriptionRuntimeSettings) -> bool:
        return worker_state.jobs_processed >= settings.max_job_per_worker

    def _is_worker_idle_too_long(self, *, worker_state: WorkerProcessState, settings: TranscriptionRuntimeSettings, now) -> bool:
        if worker_state.idle_since is None:
            return False
        idle_cutoff = now - timedelta(seconds=settings.max_idle_seconds)
        return worker_state.idle_since <= idle_cutoff

    def _get_recyclable_workers(self, *, settings: TranscriptionRuntimeSettings) -> list[tuple[str,str]]:
        from django.utils import timezone
        from user_settings.models import WorkerProcessState, WorkerStatus
        now = timezone.now()
        worker_states = {
            state.worker_id: state for state in WorkerProcessState.objects.filter(worker_id__in=self._workers.keys())
        }
        recyclable_workers: list[tuple[str, str]] = []
        for worker_id, managed in self._workers.items():
            worker_state = worker_states.get(worker_id)
            if worker_state is None:
                continue
            if worker_state.status != WorkerStatus.IDLE or worker_state.current_chunk_id is not None:
                continue
            recycle_reason: str | None = None
            if self._has_worker_reached_max_jobs(worker_state=worker_state, settings=settings):
                recycle_reason = "max_jobs_per_worker_reached"
            elif self._is_worker_idle_too_long(worker_state=worker_state, settings=settings, now=now):
                recycle_reason = "max_idle_seconds_reached"
            if recycle_reason is not None:
                logger.info(
                    "worker_supervisor_worker_marked_for_recycle",
                    extra={
                        "worker_id": worker_id,
                        "pid": managed.process.pid,
                        "status": worker_state.status,
                        "jobs_processed": worker_state.jobs_processed,
                        "idle_since": worker_state.idle_since,
                        "last_heartbeat_at": worker_state.last_heartbeat_at,
                        "recycle_reason": recycle_reason,
                        "max_jobs_per_worker": settings.max_job_per_worker,
                        "max_idle_seconds": settings.max_idle_seconds,
                    }
                )
                recyclable_workers.append((worker_id, recycle_reason))
        return recyclable_workers

    def _recycle_eligible_workers(self, *, settings: TranscriptionRuntimeSettings) -> None:
        recycle_eligible_workers: list[tuple[str, str]] = self._get_recyclable_workers(settings=settings)
        if not recycle_eligible_workers:
            return
        worker_id, reason = recycle_eligible_workers[0]
        self._stop_worker(worker_id=worker_id, reason=reason)

