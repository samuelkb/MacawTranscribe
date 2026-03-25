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


def _chunk_ownership_window(*, chunk: Chunk, total_chunks: int, overlap_ms: int) -> tuple[int, int]:
    """
    Return the global time window this chunk owns for assembly
    """
    half_overlap = overlap_ms / 2
    owned_start = chunk.start_time
    owned_end = chunk.end_time
    if chunk.chunk_index > 0:
        owned_start += half_overlap
    if chunk.chunk_index < total_chunks - 1:
        owned_end += half_overlap
    return owned_start, owned_end

def assembly_recording_transcript(*, recording: Recording, overlap_ms: int = 5_000, include_silence_annotations = True) -> dict:
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
            global_end_time = chunk.end_time + word.end_time
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
    speaker_labels = {
        label.speaker_id: label.display_name for label in SpeakerLabel.objects.filter(recording=recording)
    }
    silence_segments = list(SilenceSegment.objects.filter(recording=recording).order_by("start_time", "end_time"))
    grouped_segments: list[dict] = []

    for speaker_segment in speaker_segments:
        speaker_name = speaker_labels.get(speaker_segment.speaker_id, speaker_segment.speaker_id)
        words_in_segment = [word for word in assembled_words if word.start_time >= speaker_segment.start_time and word.start_time < speaker_segment.end_time]
        text = " ".join(word.text.strip() for word in words_in_segment if word.text.strip()).strip()
        if text:
            grouped_segments.append({
                "type":"speaker",
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
                    "type":"silence",
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
        "full_text": "\n\n".join(part for part in full_text_parts if part.strip()),
    }
