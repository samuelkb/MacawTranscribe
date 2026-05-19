import logging
from dataclasses import dataclass
from typing import Final

from recordings.models import Chunk, Recording, ChunkStatus
from speakers.models import SpeakerSegment, SpeakerLabel, SilenceSegment
from transcriptions.models import TranscriptWord

logger: Final[logging.Logger] = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledWord:
    text: str
    start_time: int
    end_time: int
    chunk_id: str


@dataclass(frozen=True)
class ChunkReviewLine:
    line_type: str
    start_time: int
    end_time: int
    speaker_id: str | None
    speaker_name: str | None
    text: str


def _chunk_ownership_window(*, chunk: Chunk, total_chunks: int, overlap_ms: int) -> tuple[int, int]:
    """
    Return the global time window this chunk owns for assembly.
    """
    half_overlap = overlap_ms / 2
    owned_start = chunk.start_time
    owned_end = chunk.end_time
    if chunk.chunk_index > 0:
        owned_start += half_overlap
    if chunk.chunk_index < total_chunks - 1:
        owned_end += half_overlap
    return owned_start, owned_end


def _build_speaker_name_map(*, recording: Recording) -> dict[str, str]:
    return {
        label.speaker_id: label.display_name
        for label in SpeakerLabel.objects.filter(recording=recording)
    }


def _load_chunk_words(*, chunk: Chunk) -> list[AssembledWord]:
    return [
        AssembledWord(
            text=word.text,
            start_time=chunk.start_time + word.start_time,
            end_time=chunk.start_time + word.end_time,
            chunk_id=str(chunk.id),
        )
        for word in TranscriptWord.objects.filter(chunk=chunk).order_by("word_index")
    ]


def _build_review_lines_for_window(
        *,
        recording: Recording,
        window_start: int,
        window_end: int,
        words: list[AssembledWord],
        include_silence_annotations: bool,
) -> list[dict[str, object]]:
    """
    Build ordered display lines for one time window from existing transcript words, speaker segments and silence rows.
    """
    if window_end <= window_start:
        return []

    speaker_name_map = _build_speaker_name_map(recording=recording)
    speaker_segments = list(
        SpeakerSegment.objects.filter(
            recording=recording,
            end_time__gt=window_start,
            start_time__lt=window_end,
        ).order_by("start_time", "end_time")
    )
    silence_segments = list(
        SilenceSegment.objects.filter(
            recording=recording,
            end_time__gt=window_start,
            start_time__lt=window_end,
        ).order_by("start_time", "end_time")
    ) if include_silence_annotations else []

    lines: list[ChunkReviewLine] = []

    for speaker_segment in speaker_segments:
        line_words = [
            word for word in words
            if word.end_time > speaker_segment.start_time and word.start_time < speaker_segment.end_time
        ]
        line_text = " ".join(word.text.strip() for word in line_words if word.text.strip()).strip()
        if not line_text:
            continue
        line_start = max(window_start, speaker_segment.start_time)
        line_end = min(window_end, speaker_segment.end_time)
        if line_end <= line_start:
            continue
        lines.append(
            ChunkReviewLine(
                line_type="speaker",
                start_time=line_start,
                end_time=line_end,
                speaker_id=speaker_segment.speaker_id,
                speaker_name=speaker_name_map.get(speaker_segment.speaker_id, speaker_segment.speaker_id),
                text=line_text,
            )
        )

    for silence_segment in silence_segments:
        line_start = max(window_start, silence_segment.start_time)
        line_end = min(window_end, silence_segment.end_time)
        if line_end <= line_start:
            continue
        duration_seconds = round((line_end - line_start) / 1000)
        lines.append(
            ChunkReviewLine(
                line_type="silence",
                start_time=line_start,
                end_time=line_end,
                speaker_id=None,
                speaker_name=None,
                text=f"[pause {duration_seconds}s]",
            )
        )

    lines.sort(key=lambda item: (item.start_time, item.end_time, item.line_type))
    return [
        {
            "type": line.line_type,
            "start_time": line.start_time,
            "end_time": line.end_time,
            "speaker_id": line.speaker_id,
            "speaker_name": line.speaker_name,
            "text": line.text,
        }
        for line in lines
    ]


def assemble_chunk_review_display(
        *,
        chunk: Chunk,
        overlap_ms: int = 5_000,
        next_chunk: Chunk | None = None,
        include_silence_annotations: bool = True,
) -> dict[str, object]:
    """
    Build interview-style display sections for one chunk review card.

    Rendering rules:
    - leading overlap is readonly context for all but the first chunk
    - the remainder of the current chunk is editable
    - optional next-chunk preview shows the first overlap window of the next chunk as readonly context
    """
    prefix_start = chunk.start_time
    prefix_end = min(chunk.start_time + overlap_ms, chunk.end_time) if chunk.chunk_index > 0 else chunk.start_time
    editable_start = prefix_end if chunk.chunk_index > 0 else chunk.start_time
    editable_end = chunk.end_time
    next_preview_start = chunk.end_time
    next_preview_end = min(chunk.end_time + overlap_ms, next_chunk.end_time) if next_chunk is not None else next_preview_start

    current_chunk_words = _load_chunk_words(chunk=chunk)
    next_chunk_words = _load_chunk_words(chunk=next_chunk) if next_chunk is not None else []

    return {
        "readonly_prefix_lines": _build_review_lines_for_window(
            recording=chunk.recording,
            window_start=prefix_start,
            window_end=prefix_end,
            words=current_chunk_words,
            include_silence_annotations=include_silence_annotations,
        ),
        "editable_lines": _build_review_lines_for_window(
            recording=chunk.recording,
            window_start=editable_start,
            window_end=editable_end,
            words=current_chunk_words,
            include_silence_annotations=include_silence_annotations,
        ),
        "readonly_next_preview_lines": _build_review_lines_for_window(
            recording=chunk.recording,
            window_start=next_preview_start,
            window_end=next_preview_end,
            words=next_chunk_words,
            include_silence_annotations=include_silence_annotations,
        ),
    }


def assembly_recording_transcript(*, recording: Recording, overlap_ms: int = 5_000, include_silence_annotations=True) -> dict:
    """
    Assemble a full-recording transcript from chunk words, speaker segments, and optional silence segments.
    :param recording: Recording object to assemble transcription
    :param overlap_ms: Overlap time in milliseconds
    :param include_silence_annotations: Whether to include silence annotations
    :return: Structured transcript payload suitable for API responses/UI rendering
    """
    chunks = list(Chunk.objects.filter(recording=recording, status=ChunkStatus.COMPLETED).order_by("chunk_index"))

    if not chunks:
        return {
            "recording_id": str(recording.id),
            "recording_status": recording.status,
            "full_text": "",
            "segments": [],
        }
    total_chunks = len(chunks)
    assembled_words: list[AssembledWord] = []

    for chunk in chunks:
        owned_start, owned_end = _chunk_ownership_window(chunk=chunk, total_chunks=total_chunks, overlap_ms=overlap_ms)
        words = list(TranscriptWord.objects.filter(chunk=chunk).order_by("word_index"))
        for word in words:
            global_start_time = chunk.start_time + word.start_time
            global_end_time = chunk.start_time + word.end_time
            if global_end_time <= owned_start:
                continue
            if global_start_time >= owned_end:
                continue
            assembled_words.append(AssembledWord(
                text=word.text,
                start_time=global_start_time,
                end_time=global_end_time,
                chunk_id=str(chunk.id),
            ))

    assembled_words.sort(key=lambda w: (w.start_time, w.end_time))
    speaker_segments = list(SpeakerSegment.objects.filter(recording=recording).order_by("start_time", "end_time"))
    speaker_labels = _build_speaker_name_map(recording=recording)
    silence_segments = list(SilenceSegment.objects.filter(recording=recording).order_by("start_time", "end_time"))
    grouped_segments: list[dict] = []

    for speaker_segment in speaker_segments:
        speaker_name = speaker_labels.get(speaker_segment.speaker_id, speaker_segment.speaker_id)
        words_in_segment = [word for word in assembled_words if word.start_time >= speaker_segment.start_time and word.start_time < speaker_segment.end_time]
        text = " ".join(word.text.strip() for word in words_in_segment if word.text.strip()).strip()
        if text:
            grouped_segments.append({
                "type": "speaker",
                "speaker_id": speaker_segment.speaker_id,
                "speaker_name": speaker_name,
                "start_time": speaker_segment.start_time,
                "end_time": speaker_segment.end_time,
                "text": text,
            })
    if include_silence_annotations:
        for silence in silence_segments:
            duration_ms = silence.end_time - silence.start_time
            duration_seconds = round(duration_ms / 1000)
            grouped_segments.append(
                {
                    "type": "silence",
                    "start_time": silence.start_time,
                    "end_time": silence.end_time,
                    "text": f"[pause {duration_seconds}s]"
                }
            )
    grouped_segments.sort(key=lambda item: (item["start_time"], item["end_time"]))

    full_text_parts: list[str] = []
    for item in grouped_segments:
        if item["type"] == "speaker":
            full_text_parts.append(f"{item['speaker_name']} \n {item['text']}")
        else:
            full_text_parts.append(item['text'])
    return {
        "recording_id": str(recording.id),
        "recording_status": recording.status,
        "full_text": "\n".join(part for part in full_text_parts if part.strip()),
        "segments": grouped_segments,
    }
