import json
import logging
from typing import Final
from uuid import UUID

from django.http import HttpRequest, JsonResponse, HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from recordings.models import Recording, RecordingStatus
from recordings.services import ingest_uploaded_recording, create_chunks

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


@csrf_exempt
@require_POST
def create_chunks_view(request: HttpRequest, recording_id: UUID) -> JsonResponse:
    """
    Create chunk metadata for a recording
    """
    try:
        recording = Recording.objects.get(id=recording_id)
    except Recording.DoesNotExist:
        return JsonResponse(
            {
                "error": "recording_not_found",
                "detail": f"Recording {recording_id} was not found.",
            },
            status=404,
        )
    chunk_duration_milliseconds = 30_000
    overlap_milliseconds = 5_000

    if request.body:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse(
                {
                    "error": "invalid_json",
                    "detail": "Request body must be a valid JSON.",
                },
                status=400,
            )
        chunk_duration_milliseconds = payload.get("duration_milliseconds", chunk_duration_milliseconds)
        overlap_milliseconds = payload.get("overlap_milliseconds", overlap_milliseconds)
    try:
        chunks = create_chunks(
            recording=recording,
            chunk_duration_milliseconds=chunk_duration_milliseconds,
            overlap_milliseconds=overlap_milliseconds,
        )
    except ValueError as exc:
        logger.warning(
            "chunk_creation_rejected",
            extra={
                "recoding_id": str(recording.id),
                "reason": str(exc),
            }
        )
        return JsonResponse(
            {
                "error": "invalid_chunk_request",
                "detail": str(exc),
            },
            status=400,
        )
    except Exception:
        logger.exception(
            "chunk_creation_failed",
            extra={
                "recoding_id": str(recording.id),
            }
        )
        return JsonResponse(
            {
                "error": "chunk_creation_failed",
                "detail": "Cunk creation failed.",
            },
            status=500,
        )
    return JsonResponse(
        {
            "recording_id": str(recording.id),
            "status": recording.status,
            "chunk_count": len(chunks),
            "chunk_duration_milliseconds": chunk_duration_milliseconds,
            "overlap_milliseconds": overlap_milliseconds,
        },
        status=201,
    )

def home_view(request: HttpRequest) -> HttpResponse:
    """
    Returns the list of recordings with status different from COMPLETED.
    """
    resume_recordings = Recording.objects.exclude(status=RecordingStatus.COMPLETED).order_by("-updated_at")
    return render(
        request,
        "recordings/pages/home.html",
        {"resume_recordings": resume_recordings},
    )


def recording_detail_view(request: HttpRequest, recording_id: UUID) -> HttpResponse:
    """
    Render the recording workspace page for a specific recording.
    :param request:
    :param recording_id: UUID of the recording to display.
    :return: Workspace page response.
    """
    recording = get_object_or_404(Recording, id=recording_id)
    return render(
        request,
        "recordings/pages/recording_detail.html",
        {"recording": recording},
    )
