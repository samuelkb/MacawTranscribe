import logging
import multiprocessing
import os
import socket
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from typing import Final, TYPE_CHECKING
from uuid import uuid4

logger: Final[logging.Logger] = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pipelines.worker import WorkerConfig
    from user_settings.models import TranscriptionRuntimeSettings

@dataclass
class ManagedWorker:
    worker_id: str
    process: multiprocessing.Process
    stop_event: MpEvent
    started_at: float


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
        from user_settings.models import TranscriptionRuntimeSettings
        logger.info("worker_supervisor_started")

        try:
            while not self._stop_requested:
                settings = TranscriptionRuntimeSettings.get_solo()
                self._reap_dead_workers()
                self._reconcile(settings=settings)
                time.sleep(settings.supervisor_poll_seconds)
        finally:
            self._shutdown_all_workers()
            logger.info("worker_supervisor_stopped")

    def _reconcile(self, *, settings: TranscriptionRuntimeSettings) -> None:
        if not settings.supervisor_enabled:
            logger.info("worker_supervisor_disabled_stopping_all_workers")
            self._scale_down_to_zero()
            return

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
        )
        logger.info(
            "worker_supervisor_spawned_worker",
            extra={
                "worker_id": worker_id,
                "pid": process.pid,
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
