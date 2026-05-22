import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Final
from unicodedata import category, normalize

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


def _normalize_overlap_word(text: str) -> str:
    """
    Normalize transcript words before overlap matching.

    Letters, combining marks and numbers remain Unicode-aware so accents used in common European languages are
    preserved. Apostrophe variants are aligned for elisions and contractions.
    """
    apostrophe_variants = {"'", "\u2018", "\u2019", "\u02bc"}
    normalized_text = normalize("NFKC", text).casefold()
    normalized_characters: list[str] = []
    for character in normalized_text:
        if character in apostrophe_variants:
            normalized_characters.append("'")
            continue
        character_category = category(character)
        if character_category[0] in {"L", "M", "N"}:
            normalized_characters.append(character)
    return "".join(normalized_characters)


def _word_intersects_window(*, word: AssembledWord, window_start: int, window_end: int) -> bool:
    return word.end_time > window_start and word.start_time < window_end


def _normalized_overlap_word_tokens(words: list[AssembledWord]) -> list[tuple[AssembledWord, str]]:
    word_tokens = [(word, _normalize_overlap_word(word.text)) for word in words]
    return [(word, token) for word, token in word_tokens if token]


def _deduplicate_matching_chunk_overlaps(
        *,
        chunk_word_groups: list[tuple[Chunk, list[AssembledWord]]],
) -> list[AssembledWord]:
    """
    Remove matched word spans from the next chunk's overlap while keeping unmatched overlap words from both chunks.
    """
    assembled_words: list[AssembledWord] = []

    for index, (chunk, words) in enumerate(chunk_word_groups):
        if index == 0:
            assembled_words.extend(words)
            continue

        previous_chunk, previous_words = chunk_word_groups[index - 1]
        overlap_start = chunk.start_time
        overlap_end = previous_chunk.end_time
        if overlap_end <= overlap_start:
            assembled_words.extend(words)
            continue

        previous_overlap_words = [
            word for word in previous_words
            if _word_intersects_window(word=word, window_start=overlap_start, window_end=overlap_end)
        ]
        current_overlap_words = [
            word for word in words
            if _word_intersects_window(word=word, window_start=overlap_start, window_end=overlap_end)
        ]
        previous_word_tokens = _normalized_overlap_word_tokens(previous_overlap_words)
        current_word_tokens = _normalized_overlap_word_tokens(current_overlap_words)
        matching_blocks = SequenceMatcher(
            a=[token for _, token in previous_word_tokens],
            b=[token for _, token in current_word_tokens],
            autojunk=False,
        ).get_matching_blocks()

        # Single-token matches in an overlap are too easy to confuse with legitimate repeated short words.
        matched_current_words = {
            current_word_tokens[token_index][0]
            for _, block_current_index, match_size in matching_blocks
            if match_size >= 2
            for token_index in range(block_current_index, block_current_index + match_size)
        }
        assembled_words.extend(word for word in words if word not in matched_current_words)

    assembled_words.sort(key=lambda word: (word.start_time, word.end_time))
    return assembled_words


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
    :param overlap_ms: Reserved overlap duration in milliseconds for recording assembly policies.
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
    chunk_word_groups = [(chunk, _load_chunk_words(chunk=chunk)) for chunk in chunks]
    assembled_words = _deduplicate_matching_chunk_overlaps(chunk_word_groups=chunk_word_groups)
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
