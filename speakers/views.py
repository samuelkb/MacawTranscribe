import json
import logging
from typing import Final
from uuid import UUID

from django.http import HttpRequest, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from recordings.models import Recording
from speakers.services import save_speaker_label

logger: Final[logging.Logger] = logging.getLogger(__name__)


@csrf_exempt
@require_POST
def save_speaker_label_view(request: HttpRequest, recording_id: UUID) -> JsonResponse:
    """
    Create or update a display label for one speaker in a recording.
    :param request:
    :param recording_id: UUID of the recording owning the speaker.
    :return: JsonResponse describing the persisted speaker label.
    """
    recording = get_object_or_404(Recording, id=recording_id)
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse(
            {
                "error": "invalid_json",
                "detail": "Request body must be a valid JSON.",
            },
            status=400,
        )

    speaker_id = payload.get("speaker_id")
    display_name = payload.get("display_name")
    try:
        speaker_label = save_speaker_label(
            recording=recording,
            speaker_id=speaker_id,
            display_name=display_name,
        )
    except ValueError as exc:
        logger.warning(
            "speaker_label_save_rejected",
            extra={
                "recording_id": str(recording.id),
                "reason": str(exc),
            }
        )
        return JsonResponse(
            {
                "error": "invalid_speaker_label_request",
                "detail": str(exc),
            },
            status=400,
        )
    except Exception:
        logger.exception(
            "speaker_label_save_failed",
            extra={
                "recording_id": str(recording.id),
                "speaker_id": speaker_id,
            }
        )
        return JsonResponse(
            {
                "error": "speaker_label_save_failed",
                "detail": "Failed to save speaker label.",
            },
            status=500,
        )

    return JsonResponse(
        {
            "recording_id": str(recording.id),
            "speaker_id": speaker_label.speaker_id,
            "display_name": speaker_label.display_name,
        },
        status=200,
    )
