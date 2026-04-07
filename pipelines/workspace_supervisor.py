import logging
import multiprocessing
import os
import socket
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from typing import Final
from uuid import uuid4

from django.conf import settings

logger: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass
class ManagedWorkspaceWorker:
    worker_id: str
    process: multiprocessing.Process
    stop_event: MpEvent
    started_at: float
    job_family: str
    queue_name: str


def _workspace_worker_entrypoint(*, worker_id: str, stop_event: MpEvent, config_dict: dict) -> None:
    """
    Child process entrypoint for the workspace pipeline worker.

    This wraps the existing worker loop and ensures fatal failures are recorded.
    :param worker_id: Worker identifier.
    :param stop_event: Event to stop worker loop.
    :param config_dict: Workspace worker configuration.
    """
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MacawTranscribe.settings")
        import django
        django.setup()
        from pipelines.workspace_worker import WorkspaceWorkerConfig, run_workspace_worker_loop
        config = WorkspaceWorkerConfig(**config_dict)
        run_workspace_worker_loop(worker_id=worker_id, config=config, stop_event=stop_event)
    except Exception as exc:
        logger.exception(
            "workspace_worker_process_fatal_failure",
            extra={
                "worker_id": worker_id,
                "error": str(exc),
            }
        )
        raise


class WorkspaceWorkerSupervisor:
    def __init__(self) -> None:
        self._workers: dict[str, ManagedWorkspaceWorker] = {}
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run_forever(self) -> None:
        from django.db import close_old_connections
        logger.info("workspace_worker_supervisor_started")

        try:
            while not self._stop_requested:
                close_old_connections()
                self._reap_dead_workers()
                self._reconcile()
                close_old_connections()
                time.sleep(getattr(settings, "WORKSPACE_WORKER_SUPERVISOR_POLL_SECONDS", 5))
        finally:
            close_old_connections()
            self._shutdown_all_workers()
            close_old_connections()
            logger.info("workspace_worker_supervisor_stopped")

    def _reconcile(self) -> None:
        target_count = max(0, int(getattr(settings, "WORKSPACE_WORKER_COUNT", 1)))
        current_count = len(self._workers)

        logger.info(
            "workspace_worker_supervisor_reconcile",
            extra={
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
        from pipelines.workspace_worker import WorkspaceWorkerConfig
        from pipelines.queue_names import build_workspace_pipeline_queue_name

        queue_name = build_workspace_pipeline_queue_name()
        worker_id = f"{socket.gethostname()}:{uuid4().hex[:12]}"
        stop_event = multiprocessing.Event()
        config = WorkspaceWorkerConfig()
        config_dict = {
            "dequeue_timeout_seconds": config.dequeue_timeout_seconds,
            "idle_sleep_seconds": config.idle_sleep_seconds,
        }
        process = multiprocessing.Process(
            target=_workspace_worker_entrypoint,
            kwargs={
                "worker_id": worker_id,
                "stop_event": stop_event,
                "config_dict": config_dict,
            },
            daemon=False,
            name=f"workspace-worker-process-{worker_id}",
        )
        process.start()
        self._workers[worker_id] = ManagedWorkspaceWorker(
            worker_id=worker_id,
            process=process,
            stop_event=stop_event,
            started_at=time.time(),
            job_family="workspace_pipeline",
            queue_name=queue_name,
        )
        logger.info(
            "workspace_worker_supervisor_spawned_worker",
            extra={
                "worker_id": worker_id,
                "pid": process.pid,
                "job_family": "workspace_pipeline",
                "queue_name": queue_name,
            }
        )

    def _stop_extra_workers(self, *, extra_count: int) -> None:
        worker_ids = list(self._workers.keys())[:extra_count]
        for worker_id in worker_ids:
            self._stop_worker(worker_id=worker_id, reason="scale_down")

    def _stop_worker(self, *, worker_id: str, reason: str) -> None:
        managed = self._workers.get(worker_id)
        if managed is None:
            return
        logger.info(
            "workspace_worker_supervisor_stopping_worker",
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
                "workspace_worker_supervisor_terminating_worker",
                extra={
                    "worker_id": worker_id,
                    "pid": managed.process.pid,
                    "reason": reason,
                }
            )
            managed.process.terminate()
            managed.process.join(timeout=5)
        self._workers.pop(worker_id, None)

    def _reap_dead_workers(self) -> None:
        dead_worker_ids: list[str] = []
        for worker_id, managed in self._workers.items():
            if managed.process.is_alive():
                continue
            exitcode = managed.process.exitcode
            logger.warning(
                "workspace_worker_supervisor_detected_dead_worker",
                extra={
                    "worker_id": worker_id,
                    "pid": managed.process.pid,
                    "exitcode": exitcode,
                }
            )
            dead_worker_ids.append(worker_id)

        for worker_id in dead_worker_ids:
            self._workers.pop(worker_id, None)

    def _shutdown_all_workers(self) -> None:
        for worker_id in list(self._workers.keys()):
            self._stop_worker(worker_id=worker_id, reason="supervisor_shutdown")
