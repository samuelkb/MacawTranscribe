import logging
import signal
from typing import Final

from django.core.management import BaseCommand

from pipelines.workspace_supervisor import WorkspaceWorkerSupervisor

logger: Final[logging.Logger] = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run the workspace pipeline worker supervisor."

    def handle(self, *args, **options) -> None:
        supervisor = WorkspaceWorkerSupervisor()

        def _handle_signal(signum, frame) -> None:
            logger.info(
                "workspace_worker_supervisor_signal_received",
                extra={"signum": signum}
            )
            supervisor.request_stop()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        supervisor.run_forever()
