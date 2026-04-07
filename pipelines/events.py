import json
import logging
import time
from collections.abc import Iterator
from typing import Final
from uuid import UUID

from django.conf import settings

from pipelines.queue import get_redis_client
from pipelines.queue_names import build_workspace_event_channel_name

logger: Final[logging.Logger] = logging.getLogger(__name__)


def publish_workspace_event(*, recording_id: UUID, event_type: str, payload: dict[str, object]) -> None:
    """
    Publish one workspace pipeline event to the recording-specific Redis channel.
    :param recording_id: Recording identifier receiving the event.
    :param event_type: SSE event type to publish.
    :param payload: Serializable event payload.
    :return: None
    """
    client = get_redis_client()
    channel_name = build_workspace_event_channel_name(recording_id=recording_id)
    message = {
        "event": event_type,
        "recording_id": str(recording_id),
        **payload,
    }

    try:
        client.publish(channel_name, json.dumps(message))
    except Exception as exc:
        logger.exception(
            "workspace_event_publish_failed",
            extra={
                "recording_id": str(recording_id),
                "event_type": event_type,
                "channel_name": channel_name,
                "error": str(exc),
            }
        )
        raise

    logger.info(
        "workspace_event_published",
        extra={
            "recording_id": str(recording_id),
            "event_type": event_type,
            "channel_name": channel_name,
        }
    )


def workspace_event_stream(*, recording_id: UUID) -> Iterator[str]:
    """
    Yield server-sent events from the recording-specific Redis channel.
    :param recording_id: Recording identifier receiving the event stream.
    :return: Iterator of SSE-formatted message chunks.
    """
    client = get_redis_client()
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    channel_name = build_workspace_event_channel_name(recording_id=recording_id)
    heartbeat_seconds = int(getattr(settings, "WORKSPACE_SSE_HEARTBEAT_SECONDS", 15))
    retry_milliseconds = int(getattr(settings, "WORKSPACE_SSE_RETRY_MILLISECONDS", 3000))

    logger.info(
        "workspace_event_stream_opened",
        extra={
            "recording_id": str(recording_id),
            "channel_name": channel_name,
            "heartbeat_seconds": heartbeat_seconds,
            "retry_milliseconds": retry_milliseconds,
        }
    )

    pubsub.subscribe(channel_name)
    last_heartbeat_at = time.monotonic()

    try:
        yield f"retry: {retry_milliseconds}\n\n"

        while True:
            message = pubsub.get_message(timeout=1.0)
            if message is not None:
                raw_data = message.get("data")
                if raw_data is None:
                    continue
                try:
                    payload = json.loads(raw_data)
                except Exception as exc:
                    logger.warning(
                        "workspace_event_stream_invalid_payload",
                        extra={
                            "recording_id": str(recording_id),
                            "channel_name": channel_name,
                            "error": str(exc),
                        }
                    )
                    continue
                event_type = payload.get("event", "message")
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(payload)}\n\n"
                last_heartbeat_at = time.monotonic()
                continue

            if time.monotonic() - last_heartbeat_at >= heartbeat_seconds:
                yield ": heartbeat\n\n"
                last_heartbeat_at = time.monotonic()
    finally:
        try:
            pubsub.close()
        finally:
            logger.info(
                "workspace_event_stream_closed",
                extra={
                    "recording_id": str(recording_id),
                    "channel_name": channel_name,
                }
            )
