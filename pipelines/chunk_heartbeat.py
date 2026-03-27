from uuid import UUID

from django.utils import timezone

from recordings.models import Chunk, ChunkStatus


def update_chunk_heartbeat(*, chunk_id: UUID, worker_id: str) -> None:
    """
    Update the heartbeat for a chunk if it is still actively processing.
    :param chunk_id: UUID of the chunk to update
    :param worker_id: Worker identifier.
    """
    Chunk.objects.filter(id=chunk_id, status=ChunkStatus.PROCESSING, worker_id=worker_id).update(
        heartbeat_at=timezone.now(),
        updated_at=timezone.now(),
    )
