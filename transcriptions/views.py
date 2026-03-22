import json
import logging
from typing import Final
from uuid import UUID

from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from ml.types import BackendName, ModelName
from transcriptions.services import transcribe_chunk, ChunkTranscriptionError

logger: Final[logging.Logger] = logging.getLogger(__name__)

def _parse_backend(value: str | None) -> BackendName | None:
    if value is None or not value.strip():
        return None
    return BackendName(value)

def _parse_model(value: str | None) -> ModelName | None:
    if value is None or not value.strip():
        return None
    return ModelName(value)

@csrf_exempt
@require_POST
def transcribe_chunk_view(request: HttpRequest, chunk_id: UUID) -> JsonResponse:
    """
    Transcribe one chunk using the configured or requested backend/model
    """
    backend: BackendName | None = None
    model: ModelName | None = None
    worker_id: str | None = None

    if request.body:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse(
                {
                    "error": "invalid_json",
                    "detail": "Request body must be valid JSON.",
                },
                status=400,
            )
        try:
            backend: BackendName = _parse_backend(payload.get("backend"))
            model: ModelName = _parse_model(payload.get("model"))
        except ValueError as exc:
            return JsonResponse(
                {
                    "error": "invalid_backend_or_model",
                    "detail": str(exc),
                },
                status=400,
            )
        worker_id = payload.get("worker_id")
    try:
        transcript = transcribe_chunk(
            chunk_id=chunk_id,
            backend=backend,
            model=model,
            worker_id=worker_id,
        )
    except ChunkTranscriptionError as exc:
        logger.warning(
            "chunk_transcription_rejected",
            extra={
                "chunk_id": str(chunk_id),
                "error": str(exc),
            }
        )
        return JsonResponse(
            {
                "error": "chunk_transcription_failed",
                "detail": str(exc),
            },
            status=400,
        )
    except Exception:
        logger.exception(
            "chunk_transcription_endpoint_failed",
            extra={
                "chunk_id": str(chunk_id),
                "backend": backend.value if backend else None,
                "model": model.value if model else None,
            }
        )
        return JsonResponse(
            {
                "error": "unexpected_error",
                "detail": "Chunk transcription failed unexpectedly.",
            },
            status=500,
        )

    transcript.refresh_from_db()
    chunk = transcript.chunk
    chunk.refresh_from_db()
    return JsonResponse(
        {
            "chunk_id": str(chunk_id),
            "recording_id": str(chunk.recording_id),
            "status": chunk.status,
            "accepted_text": transcript.accepted_text,
            "model_used": transcript.model_used,
            "has_pending_candidate": chunk.has_pending_candidate,
            "word_count": chunk.transcript_words.count(),
        },
        status=201,
    )
