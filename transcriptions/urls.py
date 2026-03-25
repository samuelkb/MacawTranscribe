from django.urls import path

from transcriptions.views import transcribe_chunk_view, full_recording_transcription_view

app_name = 'transcriptions'

urlpatterns = [
    path("chunks/<uuid:chunk_id>/transcribe/", transcribe_chunk_view, name="transcribe_chunk"),
    path("recordings/<uuid:recording_id>/full-transcript", full_recording_transcription_view, name="full_recording_transcript")
]