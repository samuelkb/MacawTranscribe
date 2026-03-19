import logging
from typing import Final

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from recordings.services import ingest_uploaded_recording

logger: Final[logging.Logger] = logging.getLogger(__name__)


@csrf_exempt
@require_POST
def upload_recording(request: HttpRequest) -> JsonResponse:
    """
    Upload a recording, persist the original file, create its initial database entry and attempt normalization.
    Expected multipart field name: file
    :param request:
    :return: A JSON payload describing the created recording
    """
    uploaded_file = request.FILES.get("file")
    if uploaded_file is None:
        return JsonResponse(
            {"error": "missing_file", "detail": "Request must include a file field named 'file'"},
            status=400
        )

    try:
        recording = ingest_uploaded_recording(uploaded_file=uploaded_file)
    except ValueError as exc:
        logger.warning(
            "recording_upload_rejected",
            extra={
                "reason": str(exc),
            }
        )
        return JsonResponse(
            {"error": "invalid_upload", "detail": str(exc)}, status=400,
        )
    except Exception as exc:
        logger.exception(
            "recording_upload_failed",
            extra={
                "uploaded_file": getattr(uploaded_file, "name", None),
            }
        )
        return JsonResponse(
            {
                "error": "uploaded_failed",
                "detail": "Recording ingestion failed",
            }, status=500,
        )

    return JsonResponse(
        {
            "recording_id": str(recording.id),
            "original_file_name": recording.original_file_name,
            "duration_milliseconds": recording.duration_milliseconds,
            "status": recording.status,
            "original_file_path": recording.original_file_path,
            "normalized_file_path": recording.normalized_file_path,
        },
        status=201,
    )
