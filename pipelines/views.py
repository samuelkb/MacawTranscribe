import logging
from typing import Final

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from pipelines.services import upload_and_normalize_recording, upload_normalize_and_diarize_recording, \
    upload_normalize_diarize_and_vad_recording

logger: Final[logging.Logger] = logging.getLogger(__name__)


def upload_recording_ui(request):
    """
    Simple developer UI to test recording uploads.
    """
    return render(request, "pipelines/upload.html")

@csrf_exempt
@require_POST
def upload_normalize_recording(request: HttpRequest) -> JsonResponse:
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
        result = upload_and_normalize_recording(uploaded_file=uploaded_file)
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

    recording = result.recording
    payload = {
            "recording_id": str(recording.id),
            "original_file_name": recording.original_file_name,
            "duration_milliseconds": recording.duration_milliseconds,
            "status": recording.status,
            "original_file_path": recording.original_file_path,
            "normalized_file_path": recording.normalized_file_path,
            "normalization_succeeded": result.normalization_succeeded,
    }
    if result.warning:
        payload["warning"] = result.warning
    return JsonResponse(payload, status=201)

@csrf_exempt
@require_POST
def upload_normalize_and_diarize_recording_view(request: HttpRequest) -> JsonResponse:
    """
    Upload a recording and run the first three pipeline stages:

    1. ingest upload
    2. normalize audio
    3. run diarization
    :param request:
    :return: A JSON payload describing the pipeline result.
    """
    uploaded_file = request.FILES.get("file")
    if uploaded_file is None:
        return JsonResponse(
            {
                "error": "missing_file",
                "detail": "Request must include a file field named 'file'",
            },
            status=400,
        )
    try:
        result = upload_normalize_and_diarize_recording(uploaded_file=uploaded_file)
    except ValueError as exc:
        logger.warning(
            "pipeline_upload_rejected",
            extra={
                "reason": str(exc),
            },
        )
        return JsonResponse(
            {
                "error": "invalid_upload",
                "detail": str(exc),
            },
            status=400,
        )
    except Exception:
        logger.exception(
            "pipeline_upload_failed",
            extra={
                "uploaded_file": getattr(uploaded_file, "name", None),
            },
        )
        return JsonResponse(
            {
                "error": "uploaded_failed",
                "detail": "Pipeline upload failed before completion",
            },
            status=500,
        )
    recording = result.recording
    payload = {
        "recording_id": str(recording.id),
        "original_file_name": recording.original_file_name,
        "duration_milliseconds": recording.duration_milliseconds,
        "status": recording.status,
        "original_file_path": recording.original_file_path,
        "normalized_file_path": recording.normalized_file_path,
        "normalization_succeeded": result.normalization_succeeded,
        "diarization_succeeded": result.diarization_succeeded,
    }
    if result.warning:
        payload["warning"] = result.warning
    return JsonResponse(payload, status=201)

@csrf_exempt
@require_POST
def upload_normalize_diarize_and_vad_recording_view(request: HttpRequest) -> JsonResponse:
    """
    Upload a recording and run the first four pipeline stages:

    1. ingest upload
    2. normalize audio
    3. run diarization
    4. VAD
    :param request:
    :return: A JSON payload describing the pipeline result.
    """
    uploaded_file = request.FILES.get("file")
    if uploaded_file is None:
        return JsonResponse(
            {
                "error": "missing_file",
                "detail": "Request must include a file field named 'file'",
            },
            status=400,
        )
    try:
        result = upload_normalize_diarize_and_vad_recording(uploaded_file=uploaded_file)
    except ValueError as exc:
        logger.warning(
            "pipeline_upload_rejected",
            extra={
                "reason": str(exc),
            }
        )
        return JsonResponse(
            {
                "error": "invalid_upload",
                "detail": str(exc),
            },
            status=400,
        )
    except Exception:
        logger.exception(
            "pipeline_upload_failed",
            extra={
                "uploaded_file": getattr(uploaded_file, "name", None),
            }
        )
        return JsonResponse(
            {
                "error": "uploaded_failed",
                "detail": "Pipeline upload failed before completion",
            },
            status=500,
        )
    recording = result.recording
    payload = {
        "recording_id": str(recording.id),
        "original_file_name": recording.original_file_name,
        "duration_milliseconds": recording.duration_milliseconds,
        "status": recording.status,
        "original_file_path": recording.original_file_path,
        "normalized_file_path": recording.normalized_file_path,
        "normalization_succeeded": result.normalization_succeeded,
        "diarization_succeeded": result.diarization_succeeded,
        "vad_succeeded": result.vad_succeeded,
    }
    if result.warning:
        payload["warning"] = result.warning

    return JsonResponse(payload, status=201)