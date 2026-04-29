import json
import logging
from pathlib import Path
import re
from collections.abc import Iterator
from typing import Final
from uuid import UUID

from django.http import HttpRequest, JsonResponse, HttpResponse, FileResponse, StreamingHttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from recordings.models import Recording, RecordingStatus
from recordings.services import ingest_uploaded_recording, create_chunks, format_duration_hhmmss

logger: Final[logging.Logger] = logging.getLogger(__name__)
RANGE_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(r"bytes=(\d*)-(\d*)")


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


def recording_list_view(request: HttpRequest) -> HttpResponse:
    """
    Render a page listing all recordings.
    :param request:
    :return: Recording list page response.
    """
    recordings = Recording.objects.all().order_by("-updated_at", "-created_at")
    recording_items: list[dict[str, object]] = [
        {
            "recording": recording,
            "formatted_duration": format_duration_hhmmss(
                duration_milliseconds=recording.duration_milliseconds,
            ),
        }
        for recording in recordings
    ]
    return render(
        request,
        "recordings/pages/recording_list.html",
        {
            "recording_items": recording_items,
            "recording_count": len(recording_items),
        },
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
        {
            "recording": recording,
            "formatted_duration": format_duration_hhmmss(
                duration_milliseconds=recording.duration_milliseconds,
            ),
        },
    )


@require_GET
def normalized_audio_view(request: HttpRequest, recording_id: UUID) -> StreamingHttpResponse | JsonResponse:
    """
    Stream the normalized audio file for a recording.
    :param request:
    :param recording_id: UUID of the recording whose normalized audio should be served.
    :return: StreamingHttpResponse streaming the normalized audio file.
    """
    recording = get_object_or_404(Recording, id=recording_id)
    if not recording.normalized_file_path or not recording.normalized_file_path.strip():
        return JsonResponse(
            {
                "error": "normalized_audio_not_ready",
                "detail": f"Recording {recording_id} does not have normalized audio yet.",
            },
            status=404,
        )

    normalized_path = Path(recording.normalized_file_path)
    if not normalized_path.exists():
        logger.warning(
            "normalized_audio_file_missing",
            extra={
                "recording_id": str(recording.id),
                "normalized_file_path": str(normalized_path),
            }
        )
        return JsonResponse(
            {
                "error": "normalized_audio_not_found",
                "detail": f"Normalized audio for recording {recording_id} was not found on disk.",
            },
            status=404,
        )

    logger.info(
        "normalized_audio_stream_started",
        extra={
            "recording_id": str(recording.id),
            "normalized_file_path": str(normalized_path),
        }
    )

    file_size = normalized_path.stat().st_size
    range_header = request.headers.get("Range")
    if not range_header:
        response = FileResponse(
            normalized_path.open("rb"),
            content_type="audio/wav",
            filename=normalized_path.name,
        )
        response["Content-Length"] = file_size
        response["Accept-Ranges"] = "bytes"
        return response

    match = RANGE_HEADER_PATTERN.fullmatch(range_header.strip())
    if match is None:
        return JsonResponse(
            {
                "error": "invalid_range_header",
                "detail": "Range header must use the bytes=start-end format.",
            },
            status=416,
        )

    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return JsonResponse(
            {
                "error": "invalid_range_header",
                "detail": "Range header must include a start or end byte.",
            },
            status=416,
        )

    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    else:
        suffix_length = int(end_text)
        start = max(file_size - suffix_length, 0)
        end = file_size - 1

    if start >= file_size or start < 0 or end < start:
        return JsonResponse(
            {
                "error": "range_not_satisfiable",
                "detail": "Requested byte range is outside the normalized audio file.",
            },
            status=416,
        )

    end = min(end, file_size - 1)
    content_length = end - start + 1

    def _range_iterator(*, file_path: Path, offset: int, length: int, chunk_size: int = 8192) -> Iterator[bytes]:
        remaining = length
        with file_path.open("rb") as file_pointer:
            file_pointer.seek(offset)
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                chunk = file_pointer.read(read_size)
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    logger.info(
        "normalized_audio_partial_stream_started",
        extra={
            "recording_id": str(recording.id),
            "normalized_file_path": str(normalized_path),
            "range_header": range_header,
            "start_byte": start,
            "end_byte": end,
            "content_length": content_length,
        }
    )

    response = StreamingHttpResponse(
        streaming_content=_range_iterator(file_path=normalized_path, offset=start, length=content_length),
        status=206,
        content_type="audio/wav",
    )
    response["Content-Length"] = content_length
    response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    response["Accept-Ranges"] = "bytes"
    return response
